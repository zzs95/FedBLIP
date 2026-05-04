# fed_server_autoloop.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import time
import math
import torch
import asyncio
import uvicorn
import threading
import logging
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Query
from fastapi.responses import Response, JSONResponse

# ============================================================
# ⚙️ 基本配置
# ============================================================
CLIENT2CID = {
    "Brown": 0,
    "INSPECT": 1,
    "JHU": 2,
}
EXPECTED_CLIENTS = set(CLIENT2CID.keys())
NUM_CLIENTS = len(CLIENT2CID)

STORE_DIR = "stage1_fed_server_store"
os.makedirs(STORE_DIR, exist_ok=True)

DEFAULT_MAX_ROUND = 999
DEFAULT_MIN_SUBMITS = len(EXPECTED_CLIENTS)
DEFAULT_ROUND_TIMEOUT = 3600

WEIGHT_KEY0 = "num_samples"
WEIGHT_KEY1a = "val_loss_prev"
WEIGHT_KEY1b = "val_loss_curr"

# ============================================================
# 🪵 Logger
# ============================================================
def setup_logger(output_dir: str, filename: str = "fed_server.log"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, filename)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.INFO)

    logging.info(f"Logging initialized. Output → {log_file}")

# ============================================================
# 🌐 FastAPI
# ============================================================
app = FastAPI(title="Federated Server (FactFusion, Per-Client)")
state_lock = threading.Lock()

state = {
    "running": True,
    "current_round": 0,
    "max_round": DEFAULT_MAX_ROUND,
    "min_submits": DEFAULT_MIN_SUBMITS,
    "round_timeout": DEFAULT_ROUND_TIMEOUT,
    "required_clients": set(EXPECTED_CLIENTS),

    "client_weights": {},
    "pending": {},
    "round_started_at": time.time(),
    "client_val_loss_hist": {},
}

# ============================================================
# 📁 Checkpoint utils
# ============================================================
def _ckpt_path(client: str, r: int):
    return os.path.join(STORE_DIR, f"{client}_round_{r}.pth")

def _latest_ckpt_round(client: str):
    prefix = f"{client}_round_"
    rounds = []
    for f in os.listdir(STORE_DIR):
        if f.startswith(prefix) and f.endswith(".pth"):
            try:
                rounds.append(int(f[len(prefix):-4]))
            except:
                pass
    return max(rounds) if rounds else None

def _load_latest_ckpts():
    with state_lock:
        rounds = []
        for c in EXPECTED_CLIENTS:
            r = _latest_ckpt_round(c)
            if r is not None:
                sd = torch.load(_ckpt_path(c, r), map_location="cpu")
                state["client_weights"][c] = sd
                rounds.append(r)
        if rounds:
            state["current_round"] = min(rounds)
            state["round_started_at"] = time.time()
            logging.info(f"🔁 Server resumed from round {state['current_round']}")
        else:
            logging.info("ℹ️ No checkpoints found, start from round 0")

_load_latest_ckpts()

# ============================================================
# 🧮 Peer score
# ============================================================
def build_fedavg_params(
    updates: Dict[str, dict],
    client_names: List[str],
    base_sd: dict,
    keys: List[str],
    *,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Build sample-size weighted FedAvg parameters.
    All clients share the same aggregated global model.
    """
    total_samples = sum(float(updates[c].get(WEIGHT_KEY0, 0.0)) for c in client_names)
    total_samples = max(total_samples, eps)

    if verbose:
        logging.info(
            f"[FedAvg] clients={client_names}, "
            f"samples={[int(updates[c].get(WEIGHT_KEY0, 0)) for c in client_names]}, "
            f"total={total_samples:.1f}"
        )

    avg_sd: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for k in keys:
            if not torch.is_tensor(base_sd[k]):
                avg_sd[k] = base_sd[k]
                continue

            if not base_sd[k].dtype.is_floating_point:
                avg_sd[k] = base_sd[k].detach().clone().cpu()
                continue

            avg_tensor = torch.zeros_like(base_sd[k], dtype=torch.float32)

            for c in client_names:
                n_c = float(updates[c].get(WEIGHT_KEY0, 0.0))
                w_c = n_c / total_samples
                avg_tensor += updates[c]["weights"][k].to(torch.float32) * w_c

            avg_sd[k] = avg_tensor.to(dtype=base_sd[k].dtype).detach().cpu()

    return avg_sd
# ============================================================
# 🔄 Auto loop
# ============================================================
async def auto_loop():
    while True:
        await asyncio.sleep(2)
        if not state["running"]:
            continue

        r = state["current_round"]
        pending = state["pending"].get(r, {})
        enough = len(pending) >= state["min_submits"]
        timeout_hit = (time.time() - state["round_started_at"]) >= state["round_timeout"]

        if enough or timeout_hit:
            do_server_merge(r)

@app.on_event("startup")
async def on_start():
    asyncio.create_task(auto_loop())
    logging.info("🚀 Auto loop started.")
    logging.info(f"📌 EXPECTED_CLIENTS = {sorted(EXPECTED_CLIENTS)}")


# ============================================================
# 🧮 do_server_merge（仅保留完整版）
# ============================================================
def do_server_merge(round_idx: int):
    """
    Server merge with standard sample-size weighted FedAvg.
    All clients receive the same aggregated global model.
    """
    with state_lock:
        updates = state["pending"].get(round_idx, {})
        if not updates:
            logging.info(f"⚠️ [FedAvg] No pending updates for round {round_idx}")
            return False

        client_names = list(updates.keys())
        total_samples = sum(float(updates[c][WEIGHT_KEY0]) for c in client_names) or 1.0

        logging.info(
            f"⏩ Starting FedAvg: round={round_idx}, "
            f"clients={client_names}, total_samples={total_samples}"
        )

        base_name = client_names[0]
        base_sd = updates[base_name]["weights"]
        keys = list(base_sd.keys())

        avg_sd = build_fedavg_params(
            updates=updates,
            client_names=client_names,
            base_sd=base_sd,
            keys=keys,
            verbose=True,
        )

        # Store the same FedAvg model for every client.
        # This keeps the existing /global?client_name=... API compatible.
        for cname in EXPECTED_CLIENTS:
            state["client_weights"][cname] = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in avg_sd.items()
            }

        new_round = round_idx + 1
        state["current_round"] = new_round
        state["pending"].pop(round_idx, None)
        state["round_started_at"] = time.time()

        _save_client_ckpts(new_round)

        logging.info(f"★ FedAvg round {round_idx} complete → enter round {new_round}")
        return True

def _client_round_filename(client_name: str, r: int) -> str:
    return os.path.join(STORE_DIR, f"{client_name}_round_{r}.pth")

def _save_client_ckpts(round_idx: int):
    for cname, wsd in state["client_weights"].items():
        path = _client_round_filename(cname, round_idx)
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in wsd.items()}, path)

# ============================================================
# 🌐 API
# ============================================================
@app.get("/status")
def status():
    with state_lock:
        r = state["current_round"]
        submitted = set(state["pending"].get(r, {}).keys())
    return {
        "current_round": r,
        "submitted": sorted(submitted),
        "waiting_for": sorted(EXPECTED_CLIENTS - submitted),
    }

@app.get("/global")
def get_global(client_name: str = Query(...)):
    with state_lock:
        if client_name not in state["client_weights"]:
            raise HTTPException(404, f"No model for {client_name}")
        sd = state["client_weights"][client_name]

    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)

    logging.info(f"📦 [/global] Send {client_name} @ round {state['current_round']}")
    return Response(buf.read(), media_type="application/octet-stream")

@app.post("/submit_update")
async def submit_update(
    client_name: str = Form(...),
    round_idx: int = Form(...),
    num_samples: int = Form(...),
    val_loss_prev: float = Form(...),
    val_loss_curr: float = Form(...),
    weights_file: UploadFile = File(...),
):
    if client_name not in EXPECTED_CLIENTS:
        raise HTTPException(400, "Unknown client")
    if round_idx != state["current_round"]:
        raise HTTPException(409, "Round mismatch")

    sd = torch.load(io.BytesIO(await weights_file.read()), map_location="cpu")

    with state_lock:
        state["pending"].setdefault(round_idx, {})
        state["pending"][round_idx][client_name] = {
            "weights": sd,
            WEIGHT_KEY0: num_samples,
            WEIGHT_KEY1a: val_loss_prev,
            WEIGHT_KEY1b: val_loss_curr,
        }

        hist = state["client_val_loss_hist"].get(client_name, [])
        hist.append(val_loss_curr)
        state["client_val_loss_hist"][client_name] = hist[-3:]

    logging.info(
        f"✓ Received {client_name} @ round {round_idx} "
        f"({len(state['pending'][round_idx])}/{len(EXPECTED_CLIENTS)})"
    )
    return {"ok": True}

# ============================================================
# 🚀 Main
# ============================================================
if __name__ == "__main__":
    LOG_DIR = "stage1_fed_logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    setup_logger(LOG_DIR, filename="fed_server.log")  # 你想存哪里都行，比如 STORE_DIR、fed_server_store 等
    uvicorn.run(app, host="0.0.0.0", port=8008)
