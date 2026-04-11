# fed_server_autoloop_factfusion.py
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
from pydantic import BaseModel

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
def _peer_score_and_delta(u: dict, gamma: float, T_delta: float, clip_delta: float, eps: float) -> Tuple[float, float, float]:
    n = float(u.get(WEIGHT_KEY0, 0.0))
    prev = float(u.get(WEIGHT_KEY1a, u.get(WEIGHT_KEY1b, 0.0)))
    curr = float(u.get(WEIGHT_KEY1b, prev))
    delta = prev - curr
    logging.info(f"delta: {delta}")
    if clip_delta is not None:
        delta = max(-clip_delta, min(clip_delta, delta))

    q = math.exp(delta / max(T_delta, eps))
    score = (max(n, 0.0) ** gamma) * q
    return score, delta, n

def build_personalized_mixed_params(
    updates: Dict[str, dict],
    client_names: List[str],
    base_sd: dict,
    keys: List[str],
    *,
    # ---------- peer score ----------
    gamma: float = 0.5,
    T_delta: float = 0.005,
    clip_delta: float = 0.02,

    # ---------- delta stabilization (NEW) ----------
    delta_saturate_s: Optional[float] = 0.01,   # 用 tanh 饱和 delta，避免 exp 爆炸；None 表示关闭

    # ---------- teacher selection ----------
    topk: int = 2,
    require_improve: bool = True,               # 基本规则：优先选 delta > teacher_delta_min
    teacher_delta_min: float = 0.0,             # teacher 的最低 delta（默认 0，即必须改善）
    allow_strong_teacher: bool = True,          # ✅ 增强 2：强中心即使 delta≈0 也允许做 teacher
    strong_teacher_quantile: float = 0.75,      # score 位于前 (1-q) 的中心视为 strong
    strong_teacher_min_score: Optional[float] = None,  # 若提供则覆盖 quantile 阈值

    # ---------- teacher size control (NEW) ----------
    n_min_teacher: Optional[int] = None,        # teacher 硬门槛（建议传入：int(0.3*max_n) 或固定 1000）
    teacher_use_size_weight: bool = True,       # teacher 权重乘 size penalty
    teacher_size_eta: float = 1.0,              # size weight 幂次：0.5~2.0
    teacher_min_size_w: float = 0.1,            # size weight 下限（避免变 0）

    # ---------- gate / target mixing ----------
    gate_mode: str = "sigmoid",                 # "sigmoid"（连续） or "fixed"
    gate_good: float = 0.10,                    # gate_mode="fixed" 时：target improved
    gate_bad: float = 0.60,                     # gate_mode="fixed" 时：target not improved
    g_min: float = 0.05,
    g_max: float = 0.85,
    T_gate: float = 0.003,                      # gate 温度：越小越“硬”，建议 0.002~0.01
    gate_center: float = 0.0,                   # gate 中心（delta=0）

    # ---------- optional: size bias for target gate ----------
    gate_use_size_bias: bool = False,
    size_bias_N0: float = 1000.0,
    size_bias_strength: float = 0.25,

    eps: float = 1e-12,
    verbose: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    为每个 target client 生成“个性化混合后的参数”（只包含 keys）。
    mixed[k][tgt] = Tensor(cpu)

    机制：
      - teacher 候选：优先 delta>0 且样本数足够；同时允许 score 很强的中心（大中心/稳定）在 delta≈0 时继续做 teacher
      - teacher 权重：按 score（含 exp(delta/T) 与 n^gamma）并乘 size weight，抑制“小中心 delta 很大”的误导
      - target gate：连续函数，delta 越负吸收越多；可选再加 size bias 让小中心吸收更多
      - delta 饱和：tanh 缓解 exp 爆炸
    """
    # --- 1) precompute score/delta/n for all clients ---
    score: Dict[str, float] = {}
    delta: Dict[str, float] = {}
    ns: Dict[str, float] = {}

    for c in client_names:
        s, d, n = _peer_score_and_delta(updates[c], gamma, T_delta, clip_delta, eps)

        # NEW: delta saturation to avoid exp explosion & small-client overfit dominance
        if delta_saturate_s is not None:
            s0 = float(delta_saturate_s)
            d = math.tanh(d / max(s0, eps)) * s0

            # re-compute score with saturated delta (important!)
            q = math.exp(d / max(T_delta, eps))
            s = (max(n, 0.0) ** gamma) * q

        score[c], delta[c], ns[c] = float(s), float(d), float(n)
        
    # teacher hard threshold (default: 0 if not set)
    max_n = max([ns[c] for c in client_names] + [1.0])
    if n_min_teacher is None:
        n_min_teacher_eff = 0
    else:
        n_min_teacher_eff = int(n_min_teacher)

    # --- 2) compute strong teacher threshold ---
    scores_sorted = sorted([score[c] for c in client_names])
    if len(scores_sorted) == 0:
        strong_thr = float("inf")
    else:
        if strong_teacher_min_score is not None:
            strong_thr = float(strong_teacher_min_score)
        else:
            q = max(0.0, min(1.0, float(strong_teacher_quantile)))
            idx = int(round((len(scores_sorted) - 1) * q))
            strong_thr = scores_sorted[idx]

    def _teacher_size_w(c: str) -> float:
        """teacher size weight: suppress small clients even if delta is large."""
        if not teacher_use_size_weight:
            return 1.0
        w = (max(ns[c], 0.0) / max(max_n, 1.0)) ** float(teacher_size_eta)
        return max(float(teacher_min_size_w), float(w))

    def _is_teacher_eligible(src: str, tgt: str) -> bool:
        if src == tgt:
            return False
        # NEW: teacher hard size gate
        if ns[src] < n_min_teacher_eff:
            return False
        
        # ✅ 限制大中心必须连续 3 次提升
        if ns[src] >= n_min_teacher_eff:
            hist = state.get("client_val_loss_hist", {}).get(src, [])
            # if len(hist) < 3 or not (hist[0] > hist[1] > hist[2]):
            if len(hist) < 3 or hist[-1] >= sum(hist[:-1]) / (len(hist) - 1): # 检查最近一次是否好于前两次均值
                logging.info(f"[Reject] {src} failed 3x improvement check. hist={hist}")
                return False
            
        d = delta[src]
        s = score[src]
            
        if require_improve:
            # normal eligible if improves enough
            if d > float(teacher_delta_min):
                return True
            # NEW: allow strong teacher even if delta≈0 (or slightly negative if teacher_delta_min<0)
            if allow_strong_teacher and (s >= strong_thr):
                return True
            return False
        else:
            return True

    def _compute_gate_for_target(tgt: str) -> float:
        d = float(delta[tgt])
        n = float(ns[tgt])

        if gate_mode == "fixed":
            g = float(gate_good) if d > 0 else float(gate_bad)
        else:
            # sigmoid: delta 越负 -> gate 越大
            z = (float(gate_center) - d) / max(float(T_gate), eps)
            sgm = 1.0 / (1.0 + math.exp(-z))
            g = float(g_min) + (float(g_max) - float(g_min)) * sgm

        # optional: size bias (small client absorbs more)
        if gate_use_size_bias:
            bias = math.sqrt(float(size_bias_N0) / (float(size_bias_N0) + max(n, 0.0) + eps))
            g = g + float(size_bias_strength) * bias

        # clamp
        g = max(float(g_min), min(float(g_max), float(g)))
        return g

    # --- 3) build personalized mixed params ---
    mixed: Dict[str, Dict[str, torch.Tensor]] = {k: {} for k in keys}

    with torch.no_grad():
        for tgt in client_names:
            # eligible teachers
            peers = [src for src in client_names if _is_teacher_eligible(src, tgt)]

            if len(peers) == 0:
                for k in keys:
                    mixed[k][tgt] = updates[tgt]["weights"][k].detach().clone().cpu()
                if verbose:
                    logging.info(f"[PERS] {tgt}: no eligible peers -> keep local (Δ_tgt={delta[tgt]:+.4f})")
                continue

            # rank by (score * size_weight)  (NEW)
            peers.sort(key=lambda c: score[c] * _teacher_size_w(c), reverse=True)
            peers = peers[:max(1, int(topk))]

            # alpha normalization over peers (NEW includes size weight)
            raw = [max(score[p], 0.0) * _teacher_size_w(p) for p in peers]
            ssum = sum(raw) + eps
            alphas = [r / ssum for r in raw]

            g = _compute_gate_for_target(tgt)

            if verbose:
                peer_str = ", ".join([
                    f"{p}(α={a:.2f},Δ={delta[p]:+.4f},n={int(ns[p])},"
                    f"score={score[p]:.2e},sw={_teacher_size_w(p):.2f})"
                    for p, a in zip(peers, alphas)
                ])
                logging.info(
                    f"[PERS] {tgt}: g={g:.2f}, Δ_tgt={delta[tgt]:+.4f}, "
                    f"n_tgt={int(ns[tgt])}, teacher_n_min={n_min_teacher_eff}, "
                    f"strong_thr={strong_thr:.2e} <- {peer_str}"
                )

            for k in keys:
                local = updates[tgt]["weights"][k].to(torch.float32)
                teacher = torch.zeros_like(local, dtype=torch.float32)
                for src, a in zip(peers, alphas):
                    teacher += updates[src]["weights"][k].to(torch.float32) * float(a)
                out = (1.0 - g) * local + g * teacher
                mixed[k][tgt] = out.to(dtype=base_sd[k].dtype).detach().cpu()

    return mixed

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
    Server merge (NO global_weights):
    - 每个 client 维护自己的完整权重：state["client_weights"][cname]
    """
    with state_lock:
        updates = state["pending"].get(round_idx, {})
        if not updates:
            logging.info(f"⚠️ [ServerMerge] No pending updates for round {round_idx}")
            return False

        client_names = list(updates.keys())
        total_samples = sum(float(updates[c][WEIGHT_KEY0]) for c in client_names) or 1.0
        logging.info(f"⏩ Starting ServerMerge: round={round_idx}, clients={client_names}, total_samples={total_samples}")

        for cname in client_names:
            wsd = updates[cname]["weights"]
            state["client_weights"][cname] = {k: v.detach().clone().cpu() for k, v in wsd.items()}

        base_name = client_names[0]
        base_sd = updates[base_name]["weights"]

        # cross_keys = [k for k in base_sd if "cross" in k or "query" in k or "style" in k or "center" in k]
        cross_keys = list(base_sd.keys())
        max_n = max(int(updates[c][WEIGHT_KEY0]) for c in client_names)
        logging.info(f'sample size: {[int(updates[c][WEIGHT_KEY0]) for c in client_names]}')
        n_min_teacher = int(0.30 * max_n)  # 只从前30%规模的中心学；可调 0.2~0.5
        logging.info(f'n_min_teacher: {n_min_teacher}')

        mixed_personal = build_personalized_mixed_params(
            updates=updates,
            client_names=client_names,
            base_sd=base_sd,
            keys=cross_keys,

            gamma=0.5,
            T_delta=0.005,
            clip_delta=0.02,
            delta_saturate_s=0.01,

            topk=2,
            require_improve=True,
            teacher_delta_min=0.0,
            allow_strong_teacher=False,
            strong_teacher_quantile=0.75,

            n_min_teacher=n_min_teacher,
            teacher_use_size_weight=True,
            teacher_size_eta=1.0,
            teacher_min_size_w=0.1,

            gate_mode="sigmoid",
            T_gate=0.003,
            g_min=0.05,
            g_max=0.85,

            verbose=True,
        )

        for cname in client_names:
            state["client_weights"][cname] = {
                k: mixed_personal[k][cname].clone() for k in mixed_personal
            }

        new_round = round_idx + 1
        state["current_round"] = new_round
        state["pending"].pop(round_idx, None)
        state["round_started_at"] = time.time()
        _save_client_ckpts(new_round)

        logging.info(f"★ ServerMerge round {round_idx} complete → enter round {new_round}")
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
