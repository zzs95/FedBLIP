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
import logging.config
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel


# ============================================================
# ⚙️ 基本配置
# ============================================================
# ✅ 稳定 CID：务必固定且覆盖你所有会参与训练的中心
CLIENT2CID = {
    "Brown": 0,
    "INSPECT": 1,
    "JHU": 2,
}
EXPECTED_CLIENTS = set(CLIENT2CID.keys())        # 预期客户端列表
NUM_CLIENTS = len(CLIENT2CID)

STORE_DIR = "stage2_fed_server_store"                   # 权重保存目录
os.makedirs(STORE_DIR, exist_ok=True)

DEFAULT_MAX_ROUND = 999
DEFAULT_MIN_SUBMITS = len(EXPECTED_CLIENTS)      # 每轮最少提交数
DEFAULT_ROUND_TIMEOUT = 3600                     # 每轮超时（秒）
WEIGHT_KEY0 = "num_samples"                       # 聚合加权键
WEIGHT_KEY1a = "val_loss_prev"                       # 聚合加权键
WEIGHT_KEY1b = "val_loss_curr"                       # 聚合加权键

SERVER_MOMENTUM = 0.95

def setup_logger(output_dir: str, filename: str = "train.log"):
    """最简：输出到文件 + 控制台（同时收集 uvicorn 日志）"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, filename)

    # 避免重复添加 handler（例如热重载/重复导入）
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    # 关键：把 uvicorn 的日志也挂到 root（否则可能不进文件）
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.INFO)

    logging.info(f"Logging initialized. Output → {log_file}")
    return log_file


# ============================================================
# 🌐 FastAPI 实例与状态锁
# ============================================================
app = FastAPI(title="Federated Server (Per-Client Weights, Auto Rounds)")
state_lock = threading.Lock() # 🔐 防止聚合、状态查询并发冲突


# ============================================================
# 🧠 全局服务器状态
# ============================================================
state = {
    "running": True,
    "current_round": 0,
    "max_round": DEFAULT_MAX_ROUND,
    "min_submits": DEFAULT_MIN_SUBMITS,
    "round_timeout": DEFAULT_ROUND_TIMEOUT,
    "required_clients": set(EXPECTED_CLIENTS),

    # ✅ 只保存每个 client 自己的完整权重（不再保存 global_weights）
    "client_weights": {},              # Dict[str, Dict[str, Tensor]]
    "client_cls_loss": {},              # Dict[str, Tensor]

    "round_started_at": time.time(),
    "pending": {},                     # {round: {client: {"weights": sd, "num_samples": n}}}
    "client_val_loss_hist": {}, 
}


# ============================================================
# 📁 工具函数：checkpoint 保存 / 恢复（每个client一份）
# ============================================================
def _client_round_filename(client_name: str, r: int) -> str:
    return os.path.join(STORE_DIR, f"{client_name}_round_{r}.pth")


def _latest_client_ckpt(client_name: str) -> Optional[int]:
    latest = None
    prefix = f"{client_name}_round_"
    for f in os.listdir(STORE_DIR):
        if f.startswith(prefix) and f.endswith(".pth"):
            try:
                ridx = int(f[len(prefix):].split(".")[0])
                latest = ridx if latest is None else max(latest, ridx)
            except:
                pass
    return latest


def _save_client_ckpts(round_idx: int):
    """保存本轮每个 client 的完整权重到磁盘"""
    for cname, wsd in state["client_weights"].items():
        path = _client_round_filename(cname, round_idx)
        torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in wsd.items()}, path)


def _load_latest_checkpoint_if_exists():
    """
    服务器重启后：从磁盘恢复每个 client 最新权重。
    current_round 取所有 client 的最小最新轮次（保证一致性）。
    """
    with state_lock:
        round_list = []
        for cname in EXPECTED_CLIENTS:
            latest = _latest_client_ckpt(cname)
            if latest is None:
                continue
            sd = torch.load(_client_round_filename(cname, latest), map_location="cpu")
            if isinstance(sd, dict):
                state["client_weights"][cname] = sd
                round_list.append(latest)

        if round_list:
            # 为了安全：取最小值，避免某些client落后/缺失
            state["current_round"] = min(round_list)
            state["round_started_at"] = time.time()
            logging.info(f"🔁 Server resumed from round {state['current_round']} (per-client checkpoints)")
        else:
            logging.info("ℹ️ No checkpoints found; start from round 0")

_load_latest_checkpoint_if_exists()
# ============================================================
# 🧮 cls-head fedavg
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
# 🧮  聚合逻辑 cls Dia-Fuse, MRG Style Decoupled
# ============================================================
def do_server_merge(round_idx: int):
    """
    Server merge (NO global_weights):
    - 每个 client 维护自己的完整权重：state["client_weights"][cname]
    - 更新：
        (A) cross-attention：个性化 selective-teacher + gate 混合（每个 client 一套）
        (B) query_tokens：同上（每个 client 一套）
        (C) style_tokens：按 client 槽位覆盖，broadcast 给所有 client
        (D) style_center：按 client 槽位覆盖，broadcast 给所有 client
        (E) classification branch（cls_embedder / classifier）：DiaFuse
    - 其他参数：保持各自 client 本地，不聚合
    """
    with state_lock:
        updates = state["pending"].get(round_idx, {})
        if not updates:
            logging.info(f"⚠️ [ServerMerge] No pending updates for round {round_idx}")
            return False

        client_names = list(updates.keys())
        total_samples = sum(float(updates[c][WEIGHT_KEY0]) for c in client_names) or 1.0
        logging.info(
            f"⏩ Starting ServerMerge: round={round_idx}, "
            f"clients={client_names}, total_samples={total_samples}"
        )

        # --- 1) 先保存每个提交 client 的完整本地权重 ---
        for cname in client_names:
            wsd = updates[cname]["weights"]
            state["client_weights"][cname] = {
                k: v.detach().clone().cpu() for k, v in wsd.items()
            }

        # 用某个 client 的结构当模板
        base_name = client_names[0]
        base_sd = updates[base_name]["weights"]

        # teacher 只从“大中心”选：动态阈值（你原来的逻辑保留）
        max_n = max(int(updates[c][WEIGHT_KEY0]) for c in client_names)
        logging.info(f"sample size: {[int(updates[c][WEIGHT_KEY0]) for c in client_names]}")
        n_min_teacher = int(0.30 * max_n)
        logging.info(f"n_min_teacher: {n_min_teacher}")

        # =========================
        # (A) style_tokens 按 client 覆盖（broadcast）
        # =========================
        style_global = None
        if "style_tokens" in base_sd:
            style_global = base_sd["style_tokens"].detach().clone().cpu()

            for cname in client_names:
                if cname not in CLIENT2CID:
                    logging.info(f"⚠️ [{cname}] not in CLIENT2CID, skip style_tokens overwrite")
                    continue
                cid = CLIENT2CID[cname]
                local_style = state["client_weights"][cname].get("style_tokens", None)
                if local_style is None:
                    logging.info(f"⚠️ [{cname}] missing style_tokens")
                    continue
                if local_style.dim() == style_global.dim() and local_style.size(0) > cid:
                    style_global[cid] = local_style[cid]
                else:
                    logging.info(
                        f"⚠️ style_tokens incompatible from {cname}: {tuple(local_style.shape)}"
                    )
        else:
            logging.info("ℹ️ No style_tokens in model.")

        # =========================
        # (B) style_center 按 client 覆盖（broadcast）
        # =========================
        center_global = None
        if "style_center" in base_sd:
            center_global = base_sd["style_center"].detach().clone().cpu()

            for cname in client_names:
                if cname not in CLIENT2CID:
                    logging.info(f"⚠️ [{cname}] not in CLIENT2CID, skip style_center overwrite")
                    continue
                cid = CLIENT2CID[cname]
                local_center = state["client_weights"][cname].get("style_center", None)
                if local_center is None:
                    logging.info(f"⚠️ [{cname}] missing style_center")
                    continue
                if local_center.dim() == center_global.dim() and local_center.size(0) > cid:
                    center_global[cid] = local_center[cid]
                else:
                    logging.info(
                        f"⚠️ style_center incompatible from {cname}: {tuple(local_center.shape)}"
                    )
        else:
            logging.info("ℹ️ No style_center in model.")

        # =========================
        # (C) classification branch 用 FedAvg
        # =========================
        cls_branch_keys = [
            k for k in base_sd.keys()
            if k.startswith("cls_embedder.") or k.startswith("classifier.")
        ]

        cls_branch_global = None
        if cls_branch_keys:
            cls_branch_global = build_fedavg_params(
                updates=updates,
                client_names=client_names,
                base_sd=base_sd,
                keys=cls_branch_keys,
                verbose=True,
            )

        if cls_branch_global:
            logging.info(
                f"✅ FedAvg classification branch keys: {sorted(list(cls_branch_global.keys()))}"
            )
        else:
            logging.info("ℹ️ No classification-branch keys aggregated by FedAvg.")

        # =========================
        # 写回：只写回“本轮提交的 client”
        # =========================
        for cname in client_names:
            wsd = state["client_weights"][cname]

            # broadcast style tokens / style_center
            if style_global is not None:
                wsd["style_tokens"] = style_global.clone()
            if center_global is not None:
                wsd["style_center"] = center_global.clone()

            # classification branch FedAvg
            if cls_branch_global is not None:
                for k in cls_branch_keys:
                    wsd[k] = cls_branch_global[k].clone()

        # 更新 round 状态
        new_round = round_idx + 1
        state["current_round"] = new_round
        state["pending"].pop(round_idx, None)
        state["round_started_at"] = time.time()

        # 保存每个 client checkpoint
        _save_client_ckpts(new_round)

        logging.info(f"★ ServerMerge round {round_idx} complete → enter round {new_round}")
        return True


# ============================================================
# 🔄 自动推进循环任务
# ============================================================
async def auto_advance_loop():
    while True:
        await asyncio.sleep(2)

        if not state["running"]:
            continue

        r = state["current_round"]
        if r >= state["max_round"]:
            continue

        pending = state["pending"].get(r, {})
        submitted = set(pending.keys())

        # 条件 1：提交数达到 min_submits
        enough = len(submitted) >= state["min_submits"]
        
        # 条件 2：时间超过 timeout
        timeout_hit = (time.time() - state["round_started_at"]) >= state["round_timeout"]

        # 满足条件则聚合推进
        if enough or timeout_hit:
            logging.info(f"⏩ Aggregating: round={r}, submitted={list(submitted)}, "
                  f"min={state['min_submits']}, timeout_hit={timeout_hit}")
            do_server_merge(r)


@app.on_event("startup")
async def on_start():
    """启动 FastAPI 时自动启动后台轮次监控任务"""
    asyncio.create_task(auto_advance_loop())
    logging.info("🚀 Server auto-loop started")
    logging.info("📌 EXPECTED_CLIENTS = " + str(sorted(list(EXPECTED_CLIENTS))))
    logging.info("📌 STORE_DIR        = " + str(STORE_DIR))
    logging.info("📌 MAX_ROUND        = "+ str(state["max_round"]))
    logging.info("📌 MIN_SUBMITS      = "+ str(state["min_submits"]))


# ============================================================
# 🌐 REST API 定义
# ============================================================

@app.get("/status")
def status():
    """查询当前全局状态（客户端周期性调用）"""

    # 使用 lock 避免读写冲突
    with state_lock:
        r = state["current_round"]
        submitted = set(state["pending"].get(r, {}).keys())

        resp = {
            "running": state["running"],
            "current_round": r,
            "max_round": state["max_round"],
            "min_submits": state["min_submits"],
            "round_timeout": state["round_timeout"],
            "required_clients": sorted(list(state["required_clients"])),
            "submitted": sorted(list(submitted)),
            "waiting_for": sorted(list(state["required_clients"] - submitted)),
            # ✅ 不再有 global_weights，改成是否已有任意 client_weights
            "has_any_client_weights": len(state.get("client_weights", {})) > 0,
            "known_clients_weights": sorted(list(state.get("client_weights", {}).keys())),
            "round_started_at": state["round_started_at"],
        }

    logging.info(f"🩵 [/status] current_round={resp['current_round']} submitted={resp['submitted']}")
    return resp


@app.get("/global")
def get_global(
    client_name: Optional[str] = Query(default=None),
    x_client_name: Optional[str] = Header(default=None),
):
    """
    ✅ 现在 /global 返回“该客户端自己的完整权重”
    调用方式：
      - /global?client_name=INSPECT
      - 或 Header: X-Client-Name: INSPECT
    """
    cname = client_name or x_client_name
    if not cname:
        return JSONResponse(
            {"ok": False, "msg": "Missing client_name. Use /global?client_name=INSPECT or header X-Client-Name."},
            status_code=400
        )

    with state_lock:
        if not state.get("client_weights"):
            logging.info("🩵 [/global] No client weights saved yet")
            return JSONResponse({"ok": False, "msg": "No client weights saved yet"}, status_code=404)

        if cname not in state["client_weights"]:
            known = sorted(list(state["client_weights"].keys()))
            logging.info(f"🩵 [/global] Unknown client '{cname}', known={known}")
            return JSONResponse({"ok": False, "msg": f"Unknown client '{cname}'. Known: {known}"}, status_code=404)

        sd = state["client_weights"][cname]

    # ✅ 全部转 CPU + detach，防止序列化卡死
    sd_cpu = {}
    for k, v in sd.items():
        sd_cpu[k] = v.detach().cpu() if torch.is_tensor(v) else v

    buf = io.BytesIO()
    torch.save(sd_cpu, buf)
    buf.seek(0)
    data_bytes = buf.getvalue()

    fname = f"{cname}_round_{state['current_round']}.pth"
    logging.info(f"📦 [/global] Send client weights: {fname} ({len(data_bytes)/1e6:.2f} MB)")

    return Response(
        content=data_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={fname}",
            "Content-Length": str(len(data_bytes)),
            "X-FL-Round": str(state["current_round"]),
            "X-FL-Client": cname,
        }
    )


@app.post("/submit_update")
async def submit_update(
    client_name: str = Form(...),
    round_idx: int = Form(...),
    num_samples: int = Form(...),
    weights_file: UploadFile = File(...),
    val_loss_prev: float = Form(...),
    val_loss_curr: float = Form(...),
):
    """客户端上传本地更新 state_dict"""
    if client_name not in EXPECTED_CLIENTS:
        raise HTTPException(400, f"Unknown client: {client_name}")

    if round_idx != state["current_round"]:
        raise HTTPException(409, f"Round mismatch (server={state['current_round']}, client={round_idx})")
    # 加载上传的模型参数
    raw_bytes = await weights_file.read()
    sd = torch.load(io.BytesIO(raw_bytes), map_location="cpu")
    if not isinstance(sd, dict):
        raise HTTPException(400, "Uploaded file is not a valid state_dict")
    # 加锁写入
    with state_lock:
        state["pending"].setdefault(round_idx, {})

        if client_name in state["pending"][round_idx]:
            raise HTTPException(409, f"Client {client_name} already submitted for this round")

        state["pending"][round_idx][client_name] = {
            "weights": sd,
            WEIGHT_KEY0: int(num_samples),
            WEIGHT_KEY1a: float(val_loss_prev),
            WEIGHT_KEY1b: float(val_loss_curr)
            
        }
        # 🔁 更新 val_loss 历史记录（最多保留最近 3 轮）
        loss_hist = state["client_val_loss_hist"].get(client_name, [])
        loss_hist.append(float(val_loss_curr))
        if len(loss_hist) > 3:
            loss_hist = loss_hist[-3:]
        state["client_val_loss_hist"][client_name] = loss_hist

    logging.info(f"✓ Received update: {client_name} @ round {round_idx} "
          f"[{len(state['pending'][round_idx])}/{len(EXPECTED_CLIENTS)}]")
    return {"ok": True}


# ============================================================
# 🧩 控制接口（调试）
# ============================================================
class ControlConfig(BaseModel):
    running: bool = True
    max_round: int = DEFAULT_MAX_ROUND
    min_submits: int = DEFAULT_MIN_SUBMITS
    round_timeout: int = DEFAULT_ROUND_TIMEOUT
    required_clients: List[str] = list(EXPECTED_CLIENTS)
    note: Optional[str] = ""


@app.post("/control")
def control(cfg: ControlConfig):
    """动态更新训练控制参数（暂停/恢复/修改超时）"""
    with state_lock:
        state["running"] = cfg.running
        state["max_round"] = cfg.max_round
        state["min_submits"] = max(1, min(cfg.min_submits, len(cfg.required_clients)))
        state["round_timeout"] = cfg.round_timeout
        state["required_clients"] = set(cfg.required_clients)

    logging.info(f"⚙️ Control updated → running={cfg.running}, max_round={cfg.max_round}")
    return {"ok": True, "state": status()}


@app.post("/advance_once")
def advance_once():
    """手动推进一次聚合（用于调试）"""
    r = state["current_round"]
    ok = do_server_merge(r)
    if not ok:
        raise HTTPException(400, "No pending updates to aggregate")
    return {"ok": True, "state": status()}


# ============================================================
# 🧹 Reset API（便于测试）
# ============================================================
@app.post("/reset")
def reset_server():
    with state_lock:
        state["pending"].clear()
        state["current_round"] = 0
        state["round_started_at"] = time.time()
        state["client_weights"].clear()
        state["client_cls_loss"].clear()

        for f in os.listdir(STORE_DIR):
            if f.endswith(".pth") and ("_round_" in f):
                os.remove(os.path.join(STORE_DIR, f))

    logging.info("🧹 Server reset complete")
    return {"ok": True}


# ============================================================
# 🚀 启动入口
# ============================================================
if __name__ == "__main__":
    
    LOG_DIR = "stage2_fed_logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    setup_logger(LOG_DIR, filename="fed_server.log")  # 你想存哪里都行，比如 STORE_DIR、fed_server_store 等
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")
