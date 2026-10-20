"""校验 ONNX 推理链与 PyTorch 原实现是否数值一致。

三道关：
  1. 预处理：ONNX 输入的张量 必须与 torchvision transform 输出逐元素一致
  2. 特征：ONNX 输出的 embedding 必须与 _embed 一致
  3. 后处理：最终 score / 热图 必须一致（含 FAISS 距离 vs numpy 距离）

任何一关对不上，说明导出或重实现有偏差，不能用于生产。

用法:
    python verify_onnx.py
    python verify_onnx.py --site 皮料相机1
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import PIL.Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

from infer_custom import LeatherInspector                     # noqa: E402
from onnx_infer import OnnxPatchCore                          # noqa: E402

DATA = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG = "C:/Users/Administrator/Desktop/NG"

CASES = {
    "皮料相机1": [f"{DATA}/皮料相机1/100_20261008_101723.jpg",
                  f"{DATA}/皮料相机1/224_20261008_102254.jpg",
                  f"{NG}/CAM1/43_20261008_101441.jpg",
                  f"{NG}/CAM1/147_20261008_101941.jpg"],
    "皮料相机2": [f"{DATA}/皮料相机2/100_20261008_101719.jpg",
                  f"{DATA}/皮料相机2/68_20261008_101552.jpg",
                  f"{NG}/CAM2/89_20261008_101648.jpg",
                  f"{NG}/CAM2/110_20261008_101740.jpg"],
}


def diff_report(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        print(f"  {name:<26} 形状不一致 {a.shape} vs {b.shape}   ✗")
        return False
    d = np.abs(a - b)
    mx = float(d.max())
    rel = mx / max(float(np.abs(a).max()), 1e-9)
    # 判据：绝对误差 < 1e-4，或相对误差 < 1e-3（高分图上绝对差会被放大）
    ok = (mx < 1e-4) or (rel < 1e-3)
    print(f"  {name:<26} max|diff|={mx:.3e}   相对={rel:.3e}   "
          f"{'✓' if ok else '✗'}")
    return ok


def verify_site(site, onnx_root):
    print(f"\n{'='*80}")
    print(f"[{site}]  ONNX vs PyTorch 数值一致性")
    print("=" * 80)

    model_dir = os.path.join(HERE, "models",
                             "cam1" if "1" in site else "cam2")
    roi = (540, 30, 1780, 1270) if "1" in site else (15, 30, 1400, 1270)

    print("\n加载两侧实现...")
    # 关键：ONNX 走 CPU 计算，所以参照系必须用 PyTorch-CPU 才是同口径。
    # 拿 CUDA 比会得到 ~2e-3 的差异，那纯粹是 CPU/CUDA 浮点累加顺序不同
    # （实测 PyTorch 自己 CPU vs CUDA 也是 2.208e-03），不是 ONNX 的误差。
    pt = LeatherInspector(model_dir, device=torch.device("cpu"), roi=roi)
    ox = OnnxPatchCore(os.path.join(onnx_root, site))

    all_ok = True
    for path in CASES[site]:
        if not os.path.exists(path):
            print(f"\n  跳过（文件不存在）: {path}")
            continue
        name = os.path.basename(path)
        print(f"\n--- {name} ---")

        # 1) 预处理
        img = PIL.Image.open(path).convert("RGB")
        t_pt = pt.transform(img).unsqueeze(0)
        t_ox = ox.preprocess(img)
        all_ok &= diff_report("① 输入张量", t_pt.numpy(), t_ox)

        # 2) 特征
        # 注意：_embed(detach=True) 返回 list of (D,)，np.asarray 后是 (N, D)
        with torch.no_grad():
            emb_pt = np.asarray(pt.model._embed(t_pt.to(pt.device)))
        emb_ox = ox.sess.run(None, {ox.in_name: t_ox})[0]
        if emb_ox.ndim == 3:
            emb_ox = emb_ox[0]
        all_ok &= diff_report("② embedding", emb_pt, emb_ox)

        # 3) patch 距离（FAISS 平方 L2 vs numpy 平方 L2）
        with torch.no_grad():
            ps_pt = pt.model.anomaly_scorer.predict([emb_pt.astype(np.float32)])[0]
        ps_ox = ox._patch_scores(emb_ox)
        all_ok &= diff_report("③ patch 距离", ps_pt, ps_ox)

        # 4) 最终分数 + 热图
        score_pt, mask_pt = pt.predict(path)
        score_ox, label, amap_ox, ms, _ = ox.infer(path)
        print(f"  {'④ 图像分数':<26} PyTorch={score_pt:.6f}   "
              f"ONNX={score_ox:.6f}   差={abs(score_pt-score_ox):.3e}")
        ok4 = abs(score_pt - score_ox) < 1e-3
        all_ok &= ok4
        print(f"  {'⑤ 热图':<26} ", end="")
        all_ok &= diff_report("", mask_pt.astype(np.float32), amap_ox)
        print(f"  判定: {'NG' if score_ox >= ox.threshold else 'OK'}"
              f"  (阈值 {ox.threshold})")

    print(f"\n{'='*80}")
    print(f"[{site}] 结论: {'全部一致 ✓' if all_ok else '存在偏差 ✗'}")
    print("=" * 80)
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None)
    ap.add_argument("--onnx_root", default="models_onnx")
    args = ap.parse_args()

    sites = [args.site] if args.site else list(CASES)
    results = {s: verify_site(s, os.path.join(HERE, args.onnx_root)) for s in sites}

    print(f"\n{'='*80}")
    for s, ok in results.items():
        print(f"  {s}: {'通过 ✓' if ok else '未通过 ✗'}")
    print("=" * 80)
