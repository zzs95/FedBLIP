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
import numpy as np
from prompts.rewrite_process_ollama import LLM_pipline
from datasets_utils.abnormality_list_56 import abnormality_dict
from prompts.writing_prompts import write_findings_prompt as WRITE_FINDINGS_PROMPT

f_name = 'study_findings_pred'
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

def build_prompt(pred_study_captions,  pred_study_probs=None, write_prompt='') -> str:
    """
    pred_study_captions: list of per-scan captions (each is list[str] of length 56)
    """
    qs = ""
    abn_num = 0
    for organ in abnormality_dict.keys():
        organ_abn_len = len(abnormality_dict[organ])
        pred_abn_text_organ = []
        if pred_study_probs == None:
            for pred_image_caption in pred_study_captions:
                pred_abn_text_organ += pred_image_caption[abn_num:abn_num + organ_abn_len]   
        else:
            for pred_image_caption, pred_image_prob in zip(pred_study_captions, pred_study_probs):
                image_organ_text = pred_image_caption[abn_num:abn_num + organ_abn_len]
                image_organ_prob = pred_image_prob[abn_num:abn_num + organ_abn_len]
                image_organ_prob = np.array(image_organ_prob, dtype=np.float32)
                mask = image_organ_prob > 0.6
                image_organ_abn_text = []
                for text, m in zip(image_organ_text, mask):
                    if m:
                        image_organ_abn_text.append(text.replace('no abnormal', 'abnormal') )
                # print(image_organ_abn_text)
                
                if len(image_organ_abn_text) == 0:
                    image_organ_abn_text = ['no abnormal']
                pred_abn_text_organ += image_organ_abn_text
        abn_num += organ_abn_len

        # remove "no abnormal" lines
        pred_list = copy.deepcopy(pred_abn_text_organ)
        for s in pred_abn_text_organ:
            if "no abnormal" in s:
                try:
                    pred_list.remove(s)
                except ValueError:
                    pass

        if len(pred_list) == 0:
            if organ == "Pulmonary arteries":
                pred_list = ["No pulmonary emboli are identified."]
            elif organ == "Chest Bones":
                pred_list = ["No acute abnormality."]
            else:
                pred_list = ["Normal."]

        qs += f"Descriptions of potential abnormalities in {organ} are {pred_list}\n"

    qs += write_prompt
    return qs


def load_template_and_prompt(setname: str) -> str:
    write_prompt = WRITE_FINDINGS_PROMPT
    if setname == "Brown":
        from prompts.BUH_template import findings_template
        write_prompt = write_prompt.replace("<TEMPLATE>", findings_template)
    elif setname == "INSPECT":
        from prompts.BUH_template import findings_template
        write_prompt = write_prompt.replace("<TEMPLATE>", findings_template)
    elif setname == "JHU":
        from prompts.JHU_template import findings_template
        write_prompt = write_prompt.replace("<TEMPLATE>", findings_template)
    else:
        raise ValueError(f"Unknown setname: {setname}")
    return write_prompt


def load_abntext_df(setname: str) -> pd.DataFrame:
    if setname == "Brown":
        abntext_path = "/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/LLM_chest_section_gptoss_buh/brown_CTPA_report_sections_chest.xlsx"
    elif setname == "INSPECT":
        abntext_path = "/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/LLM_chest_section_gptoss_inspect/inspect_CTPA_report_sections_chest.xlsx"
    elif setname == "JHU":
        abntext_path = "/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/LLM_chest_section_gptoss_jhu/jhu_CTPA_report_sections_chest.xlsx"
    else:
        raise ValueError(f"Unknown setname: {setname}")
    return pd.read_excel(abntext_path)


def merge_parts(out_dir: str, ordered_ids: list, merged_name=f"{f_name}.jsonl"):
    merged_path = os.path.join(out_dir, merged_name)

    part_files = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(f"{f_name}.part") and f.endswith(".jsonl")
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

    result_df = pd.read_json(results_json, lines=True)
    abn_text_df = load_abntext_df(setname)
    
    result_df = result_df[result_df['AccessionNumber_md5'].isin(abn_text_df['AccessionNumber_md5'])] # for mix

    abn_text_df = abn_text_df[abn_text_df["AccessionNumber_md5"].isin(study_ids)]
    abn_text_df["AccessionNumber_md5"] = pd.Categorical(
        abn_text_df["AccessionNumber_md5"], categories=study_ids, ordered=True
    )
    abn_text_df = abn_text_df.sort_values("AccessionNumber_md5").reset_index(drop=True)

    write_prompt = load_template_and_prompt(setname)

    part_path = os.path.join(out_dir, f"{f_name}.part{rank}.jsonl")
    if os.path.exists(part_path):
        os.remove(part_path)

    with open(part_path, "w", encoding="utf-8") as f:
        it = tqdm(abn_text_df.iterrows(), total=len(abn_text_df), desc=f"[ollama-gpu {gpu_id}] part{rank}")
        for _, study_row in it:
            study_id = study_row["AccessionNumber_md5"]
            gen_text_df = result_df[result_df["AccessionNumber_md5"] == study_id]
            img_num = len(gen_text_df)

            pred_study_captions = [gen_text_df["abn_text"].iloc[i] for i in range(img_num)]
            pred_study_probs = [gen_text_df["abn_probs"].iloc[i] for i in range(img_num)]
            if 'AbnProbsFilter' in f_name:
                qs = build_prompt(pred_study_captions, pred_study_probs, write_prompt)
            else:
                qs = build_prompt(pred_study_captions, None, write_prompt)

            findings_text_rewrite = model_LLM.forward(qs)
            findings_text_rewrite = normalize_common_unicode(findings_text_rewrite)

            findings_gt = study_row.get("Findings Text", "")

            out_dict = {
                "AccessionNumber_md5": str(study_id),
                "findings_image_gt": str(findings_gt),
                "findings_image_pred": findings_text_rewrite,
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
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="ollama gpu ids, e.g. 2,3,4")
    parser.add_argument("--exp_path", type=str, default="./stage2_test_result/round_50")
    parser.add_argument("--results_file", type=str, default="output_results_abn_text_probs.jsonl")

    parser.add_argument("--model_id", type=str, default="gpt-oss:20b")
    # parser.add_argument("--model_id", type=str, default="llama3.1:8b")
    args = parser.parse_args()

    out_dir = os.path.join(args.exp_path, args.setname)
    # out_dir = os.path.join(args.exp_path, args.setname, args.model_id)
    os.makedirs(out_dir, exist_ok=True)

    results_json = os.path.join(out_dir, args.results_file)
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
        merged = merge_parts(out_dir, ordered_ids, merged_name=f"{f_name}.jsonl")
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

    merged = merge_parts(out_dir, ordered_ids, merged_name=f"{f_name}.jsonl")
    print(f"🎉 merged saved: {merged}")


if __name__ == "__main__":
    main()
