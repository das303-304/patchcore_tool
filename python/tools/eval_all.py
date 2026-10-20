#!/usr/bin/env python
"""批量评测 MVTec AD 全 15 子类 —— 对指定模型组。

用法:
    python eval_all.py                 # 评测全部 6 组
    python eval_all.py <模型组名>       # 评测指定组
    python eval_all.py <模型组名> <子类列表逗号分隔>

不用 bash 脚本是因为沙箱 Git Bash 下位置参数/数组会被环境变量污染，
Python 传参更可控。

注意:
  - 本机是 faiss-cpu，绝不能传 --faiss_on_gpu（会 AttributeError）
  - Windows 用 --num_workers 0（默认 8 走 spawn，重新 import torch 极慢）
"""
import os
import subprocess
import sys

ROOT = r"C:/Users/Administrator/Desktop/PatchCore"
PROJ = os.path.join(ROOT, "patchcore-inspection")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
DATA = os.path.join(ROOT, "mvtec_anomaly_detection")

DATASETS = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]

ALL_GROUPS = [
    "IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1",
    "IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1",
    "IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1",
    "IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3",
    "IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1",
    "IM320_Ensemble_L2-3_P001_D1024-384_PS-5_AN-5",
]


def img_size(group):
    """IM224 -> 256/224, IM320 -> 366/320。"""
    if group.startswith("IM320"):
        return "366", "320"
    return "256", "224"


def eval_group(group, datasets=None):
    datasets = datasets or DATASETS
    resize, imagesize = img_size(group)
    out = os.path.join(ROOT, "eval_results", group)
    os.makedirs(out, exist_ok=True)

    cmd = [PY, "bin/load_and_evaluate_patchcore.py",
           "--gpu", "0", "--seed", "0", out, "patch_core_loader"]
    for d in datasets:
        cmd += ["-p", f"models/{group}/models/mvtec_{d}"]
    cmd += ["dataset", "--resize", resize, "--imagesize", imagesize,
            "--num_workers", "0"]
    for d in datasets:
        cmd += ["-d", d]
    cmd += ["mvtec", DATA]

    env = dict(os.environ, PYTHONPATH="src")
    print(f"\n{'='*66}\n{group}  (resize={resize} imagesize={imagesize}, "
          f"{len(datasets)} 个子类)\n{'='*66}", flush=True)

    proc = subprocess.run(cmd, cwd=PROJ, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")

    for line in proc.stdout.splitlines():
        if any(k in line for k in ("Evaluating dataset", "instance_auroc",
                                   "full_pixel_auroc", "anomaly_pixel_auroc",
                                   "Traceback", "Error")):
            print("  " + line.strip(), flush=True)

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        print(f"  !! 失败 rc={proc.returncode}\n{tail}", flush=True)

    csv = os.path.join(out, "results.csv")
    if os.path.exists(csv):
        print(f"  --> {csv}", flush=True)
    return csv


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        groups = ALL_GROUPS
    elif args[0] == "all":
        groups = ALL_GROUPS
    else:
        groups = [args[0]]

    datasets = DATASETS
    if len(args) > 1:
        datasets = [d.strip() for d in args[1].split(",") if d.strip()]

    for g in groups:
        eval_group(g, datasets)

    print("\n############ 全部完成 ############")
