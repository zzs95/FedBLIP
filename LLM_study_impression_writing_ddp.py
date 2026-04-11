#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import copy
import argparse
import traceback
import multiprocessing as mp
import subprocess
import time
import pandas as pd
from tqdm import tqdm
from prompts.rewrite_process_ollama import LLM_pipline
from prompts.writing_prompts import write_impression_prompt
from datasets_utils.img_cls import anno_files
f_name = 'study_impression_pred'
# =========================
# Utils
# =========================
def normalize_common_unicode(s: str) -> str:
    s = s.replace("\u202f", " ")          # 窄 NBSP → 空格
    s = s.replace("\u00ad", "")           # soft hyphen → 删除
    s = re.sub(r"[\u2011\u2013]", "-", s) # 各类连字符 → '-'
    s = re.sub(r"[\u2019]", "'", s)       # 单引号
    s = re.sub(r"[\u201d]", '"', s)       # 双引号
    s = s.replace("\u2026", "…")        # 省略号
    return s

def restart_ollama_gpu(gpu_id: str):
    cname = f"ollama_python_gpu{gpu_id}"
    subprocess.run(["docker", "stop", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "start", cname], check=True)
    time.sleep(3)  # 给 ollama 启动一点时间

def build_prompt(pred_study_captions, write_prompt: str) -> str:
    qs = write_prompt.replace('<FINDINGS_TEXT>', pred_study_captions)
    return qs


def load_template_and_prompt(setname: str) -> str:
    write_prompt = write_impression_prompt
    if setname == "Brown":
        from prompts.BUH_template import impression_template
        write_prompt = write_prompt.replace("<TEMPLATE>", impression_template)
    elif setname == "INSPECT":
        from prompts.BUH_template import impression_template
        write_prompt = write_prompt.replace("<TEMPLATE>", impression_template)
    elif setname == "JHU":
        from prompts.JHU_template import impression_template
        write_prompt = write_prompt.replace("<TEMPLATE>", impression_template)
    else:
        raise ValueError(f"Unknown setname: {setname}")
    return write_prompt



def merge_parts(out_dir: str, ordered_ids: list, merged_name="study_impression_noAbnProbs"):
    merged_path = os.path.join(out_dir, merged_name+'.jsonl')

    part_files = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(f"{merged_name}.part") and f.endswith(".jsonl")
    ])
    if len(part_files) == 0:
        raise RuntimeError(f"No part files found in {out_dir}")

    recs = {}
    for p in part_files:
        with open(p, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                obj = json.loads(line)
                recs[obj["AccessionNumber_md5"]] = obj

    with open(merged_path, "w", encoding="utf-8") as fout:
        for sid in ordered_ids:
            sid = str(sid)
            if sid not in recs:
                fout.write(json.dumps({"AccessionNumber_md5": sid, "missing": True}, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(recs[sid], ensure_ascii=False) + "\n")

    return merged_path


# =========================
# Worker
# =========================
def worker_process(rank: int,
                   gpu_id: str,
                   study_ids: list,
                   results_json: str,
                   setname: str,
                   out_dir: str,
                   model_id: str):

    # ollama 端口映射：gpu_id='2' -> http://localhost:11437
    restart_ollama_gpu(gpu_id=str(gpu_id))
    model_LLM = LLM_pipline(model_id=model_id, gpu=str(gpu_id))


    pred_findings_df = pd.read_json(results_json, lines=True)
    result_df = pred_findings_df
    data_split, abntext_path, image_root = anno_files(setname)
    abn_text_df = pd.read_excel(abntext_path)
    
    abn_text_df = abn_text_df[abn_text_df["AccessionNumber_md5"].isin(study_ids)]
    abn_text_df["AccessionNumber_md5"] = pd.Categorical(
        abn_text_df["AccessionNumber_md5"], categories=study_ids, ordered=True
    )
    abn_text_df = abn_text_df.sort_values("AccessionNumber_md5").reset_index(drop=True)


    write_prompt = load_template_and_prompt(setname)
    
    part_path = os.path.join(out_dir, f"study_impression_noAbnProbs.part{rank}.jsonl")
    if os.path.exists(part_path):
        os.remove(part_path)
    with open(part_path, "w", encoding="utf-8") as f:
        it = tqdm(abn_text_df.iterrows(), total=len(abn_text_df), desc=f"[ollama-gpu {gpu_id}] part{rank}")
        for _, study_row in it:
            study_id = study_row["AccessionNumber_md5"]
            gen_text_df = result_df[result_df["AccessionNumber_md5"] == study_id]
            img_num = len(gen_text_df)

            pred_study_findings = gen_text_df.iloc[0]["findings_image_pred"]
            qs = build_prompt(pred_study_findings, write_prompt)

            impression_text = model_LLM.forward(qs)
            impression_text = normalize_common_unicode(impression_text)

            impression_texts_gt = study_row.get("Impression Text", "")

            out_dict = {
                "AccessionNumber_md5": str(study_id),
                "impression_image_gt": str(impression_texts_gt),
                "impression_image_pred": impression_text,
            }
            f.write(json.dumps(out_dict, ensure_ascii=False) + "\n")

    return part_path


def _run_worker(rank: int,
                gpu: str,
                ids: list,
                results_json: str,
                setname: str,
                out_dir: str,
                model_id: str):
    """
    顶层函数：spawn 可 pickle
    """
    err_log = os.path.join(out_dir, f"worker_err.part{rank}.log")
    try:
        part = worker_process(
            rank=rank,
            gpu_id=gpu,
            study_ids=ids,
            results_json=results_json,
            setname=setname,
            out_dir=out_dir,
            model_id=model_id,
        )
        # 成功时也写一下日志
        with open(err_log, "w", encoding="utf-8") as f:
            f.write(f"OK: {part}\n")
    except Exception:
        tb = traceback.format_exc()
        with open(err_log, "w", encoding="utf-8") as f:
            f.write(tb)
        # 让子进程退出码非0，主进程可检测
        raise


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setname", type=str, default="INSPECT", choices=["Brown", "INSPECT", "JHU"])
    
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6", help="ollama gpu ids, e.g. 2,3,4")
    
    parser.add_argument("--exp_path", type=str, default="./stage2_testing/round_50")
    
    parser.add_argument("--findings_results_file", type=str, default="study_findings_pred.jsonl")
    parser.add_argument("--model_id", type=str, default="gpt-oss:20b")
    # parser.add_argument("--model_id", type=str, default="llama3.1:8b")
    args = parser.parse_args()

    out_dir = os.path.join(args.exp_path, args.setname)
    os.makedirs(out_dir, exist_ok=True)

    results_json = os.path.join(out_dir, args.findings_results_file)
    if not os.path.exists(results_json):
        raise FileNotFoundError(f"results_json not found: {results_json}")

    # 全局顺序
    result_df = pd.read_json(results_json, lines=True)
    ordered_ids = result_df["AccessionNumber_md5"].unique().tolist()

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    n = len(gpu_list)
    if n == 0:
        raise ValueError("No GPUs provided.")

    groups = [ordered_ids[i::n] for i in range(n)]

    print("================================================")
    print(f"setname     = {args.setname}")
    print(f"exp_path    = {args.exp_path}")
    print(f"out_dir     = {out_dir}")
    print(f"results     = {results_json}")
    print(f"gpus        = {gpu_list}  (num_procs={n})")
    print(f"model_id    = {args.model_id}")
    print(f"num_studies = {len(ordered_ids)}")
    print("================================================")

    # ---- 单卡：不启用 mp（避免 spawn/pickle 开销与问题）
    if n == 1:
        part = worker_process(
            rank=0,
            gpu_id=gpu_list[0],
            study_ids=ordered_ids,
            results_json=results_json,
            setname=args.setname,
            out_dir=out_dir,
            model_id=args.model_id,
        )
        print(f"✅ single worker done: {part}")
        merged = merge_parts(out_dir, ordered_ids, merged_name="study_impressions_noAbnProbs.jsonl")
        print(f"🎉 merged saved: {merged}")
        return

    # ---- 多卡：spawn 多进程
    ctx = mp.get_context("spawn")
    procs = []

    for rank, (gpu, ids) in enumerate(zip(gpu_list, groups)):
        p = ctx.Process(
            target=_run_worker,
            args=(rank, gpu, ids, results_json, args.setname, out_dir, args.model_id)
        )
        p.start()
        procs.append(p)

    ok = True
    for rank, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            ok = False
            err_log = os.path.join(out_dir, f"worker_err.part{rank}.log")
            print(f"❌ worker {rank} failed (exitcode={p.exitcode}). See: {err_log}")
        else:
            print(f"✅ worker {rank} finished.")

    if not ok:
        raise RuntimeError("Some workers failed. Check worker_err.part*.log in out_dir.")

    merged = merge_parts(out_dir, ordered_ids, merged_name=f_name)
    print(f"🎉 merged saved: {merged}")


if __name__ == "__main__":
    main()
