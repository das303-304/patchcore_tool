#!/usr/bin/env python
"""在自建数据集上训练 PatchCore（无训练检测 —— 只建内存库）。

用法:
    # 最简：只给正常样本目录
    python train_own.py <正常样本目录> [模型输出目录]

    # 带缺陷样本做自评（可选）
    python train_own.py <数据根目录> <输出目录> --eval

数据目录约定（MVTec 风格，classname 可自定义）:
    <数据根>/<类名>/train/good/*.png     ← 必须：正常样本
    <数据根>/<类名>/test/good/*.png      ← 可选：留出正常样本做自评
    <数据根>/<类名>/test/<缺陷名>/*.png  ← 可选：缺陷样本
    <数据根>/<类名>/ground_truth/<缺陷名>/*.png  ← 可选：缺陷掩码

注意:
    - 本机 faiss-cpu，绝不加 --faiss_on_gpu
    - Windows 必须 --num_workers 0
    - click chain 模式下选项必须排在位置参数之前
"""
import argparse
import os
import subprocess
import sys

ROOT = r"C:/Users/Administrator/Desktop/PatchCore"
PROJ = os.path.join(ROOT, "patchcore-inspection")
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")


def run(cmd, quiet_filter=True):
    env = dict(os.environ, PYTHONPATH="src")
    print("$ " + " ".join(cmd[2:8]) + " ...", flush=True)
    p = subprocess.run(cmd, cwd=PROJ, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        if quiet_filter and ("it/s]" in s or s.endswith("%")):
            continue
        if any(k in s for k in ("Error", "Traceback", "error", "auroc", "INFO:__main__",
                                "Computing", "Subsampling", "Training", "Embedding",
                                "Saving", "FileNotFound")):
            print("  " + s, flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", help="数据根目录（其下应有 <类名>/train/good/）")
    ap.add_argument("model_out", nargs="?", default=None, help="模型输出目录")
    ap.add_argument("--classname", default=None, help="类名（默认取 data_root 下的子目录名）")
    ap.add_argument("--resize", default="256")
    ap.add_argument("--imagesize", default="224")
    ap.add_argument("--backbone", default="wideresnet50")
    ap.add_argument("--percentage", default="0.1", help="coreset 采样比例")
    ap.add_argument("--eval", action="store_true", help="训练后顺便评测 test/")
    a = ap.parse_args()

    data_root = a.data_root
    # 自动推断类名
    if a.classname:
        classname = a.classname
    else:
        subs = [d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))]
        if len(subs) == 1:
            classname = subs[0]
        else:
            print(f"!! 需要 --classname，data_root 下有多个子目录: {subs}")
            sys.exit(1)

    train_dir = os.path.join(data_root, classname, "train", "good")
    if not os.path.isdir(train_dir):
        print(f"!! 未找到正常样本目录: {train_dir}")
        sys.exit(1)
    n_train = len([f for f in os.listdir(train_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))])
    print(f"=== 类名: {classname} | 正常样本: {n_train} 张 ===")
    if n_train < 30:
        print("!! 警告: 正常样本少于 30 张，内存库覆盖不足，误报会很高")

    model_out = a.model_out or os.path.join(ROOT, f"model_{classname}")

    # 训练（建内存库）
    # 注意 click chain 模式的参数顺序：全局选项 --gpu/--seed 必须在 RESULTS_PATH 之前
    cmd = [PY, "bin/run_patchcore.py",
           "--gpu", "0", "--seed", "0", "--save_patchcore_model",
           model_out,
           "patch_core", "-b", a.backbone,
           "-le", "layer2", "-le", "layer3",
           "--pretrain_embed_dimension", "1024",
           "--target_embed_dimension", "1024",
           "--anomaly_scorer_num_nn", "1", "--patchsize", "3",
           "sampler", "-p", a.percentage, "approx_greedy_coreset",
           "dataset", "-d", classname,
           "--resize", a.resize, "--imagesize", a.imagesize,
           "--num_workers", "0",
           "mvtec", data_root]

    print("\n### 阶段一：建内存库（无梯度训练）")
    rc = run(cmd)
    if rc != 0:
        print("!! 训练失败")
        sys.exit(rc)

    # 找到生成的 run 目录（create_storage_folder 会加 _0 _1 后缀）
    proj_dir = os.path.join(model_out, "mvtec")
    if not os.path.isdir(proj_dir):
        print(f"!! 未找到输出: {proj_dir}")
        sys.exit(1)
    runs = sorted(os.listdir(proj_dir))
    run_dir = os.path.join(proj_dir, runs[-1])
    saved = os.path.join(run_dir, "models", f"mvtec_{classname}")
    print(f"\n### 内存库已保存: {saved}")
    for f in sorted(os.listdir(saved)):
        sz = os.path.getsize(os.path.join(saved, f)) / 1024
        print(f"    {f}  ({sz:.1f} KB)")

    # 可选评测
    test_dir = os.path.join(data_root, classname, "test")
    if a.eval and os.path.isdir(test_dir):
        print(f"\n### 阶段二：评测 test/")
        cmd2 = [PY, "bin/load_and_evaluate_patchcore.py",
                "--gpu", "0", "--seed", "0",
                os.path.join(ROOT, "eval_own", classname),
                "patch_core_loader", "-p", saved,
                "dataset", "-d", classname,
                "--resize", a.resize, "--imagesize", a.imagesize,
                "--num_workers", "0",
                "mvtec", data_root]
        run(cmd2)
    elif a.eval:
        print(f"\n(跳过评测：{test_dir} 不存在)")

    print(f"\n### 完成。模型路径:\n{saved}")


if __name__ == "__main__":
    main()
