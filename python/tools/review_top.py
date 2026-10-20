"""交叉验证打分 + 可疑图复核热图。

为什么这么做：
  「用整批建库再评自己」的分数是偏低的（图自己在库里）。
  所以用 2 折交叉：A建库评B、B建库评A，
  每张图都由「没见过它」的库评分 —— 这才是无偏分数。

然后按分数排序，取最可疑的若干张出热图，供人工复核：
  如果这些图人眼看着确实正常 -> 阈值有足够余量，可以上线
  如果某张其实有缺陷      -> 说明用户把它误标成 OK 了，需要修正标注
"""
import os
import sys
import argparse

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

from validate_ok_batch import (DATA_ROOT, ROI, build_bank, DEV, MEAN, STD)

OUT_DIR = os.path.join(HERE, "testimgs", "review_top")


def cv_scores(cam, resize=256, imagesize=224, pct=0.1, topk=8):
    """2 折交叉打分，返回 (文件名, 分数, 掩码) 列表（按分数降序）。"""
    d = os.path.join(DATA_ROOT, cam)
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
    roi = ROI[cam]

    A = [os.path.join(d, f) for i, f in enumerate(fs) if i % 2 == 0]
    B = [os.path.join(d, f) for i, f in enumerate(fs) if i % 2 == 1]

    tf = T.Compose([T.Lambda(lambda im: im.crop(roi)), T.Resize(resize),
                    T.CenterCrop(imagesize), T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD)])

    results = {}
    for build, hold in [(A, B), (B, A)]:
        m = build_bank(build, resize, imagesize, roi, pct)
        for p in hold:
            t = tf(PIL.Image.open(p).convert("RGB")).unsqueeze(0)
            s, mask = m.predict(t)
            results[os.path.basename(p)] = (float(s[0]), mask[0])

    # 补上没被评到的（若有）
    for f in fs:
        if f not in results:
            results[f] = (float("nan"), None)

    ranked = sorted(results.items(), key=lambda x: -x[1][0])
    return ranked, roi


def make_heatmap(img_path, mask, roi, out_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    # matplotlib 默认字体不含中文，必须显式指定，否则标题显示成方框
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    img = PIL.Image.open(img_path).convert("RGB")
    full = img
    crop = img.crop(roi)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(full); axes[0].set_title("原图（含背景）", fontsize=9)
    axes[1].imshow(crop); axes[1].set_title("ROI 裁剪", fontsize=9)
    axes[2].imshow(mask, cmap="jet"); axes[2].set_title("异常热图", fontsize=9)
    axes[3].imshow(crop.resize(mask.shape[::-1]))
    axes[3].imshow(mask, cmap="jet", alpha=0.55)
    axes[3].set_title(title, fontsize=9)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=92)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--camera", default=None)
    args = ap.parse_args()

    cams = [args.camera] if args.camera else list(ROI)
    os.makedirs(OUT_DIR, exist_ok=True)

    for cam in cams:
        print(f"\n{'='*90}")
        print(f"[{cam}] 2折交叉验证打分（每张图由没见过它的库评分）")
        print(f"{'='*90}")

        ranked, roi = cv_scores(cam)
        arr = np.array([s for _, (s, _) in ranked if not np.isnan(s)])
        print(f"  无偏分数分布: n={len(arr)}  均值 {arr.mean():.3f}  "
              f"std {arr.std():.3f}  p95 {np.percentile(arr,95):.3f}  "
              f"p99 {np.percentile(arr,99):.3f}  最大 {arr.max():.3f}")
        print(f"\n  最可疑的 {args.topk} 张:")

        for i, (name, (s, mask)) in enumerate(ranked[:args.topk]):
            print(f"    {i+1:2d}. {name:<32} {s:7.3f}")
            if mask is not None:
                make_heatmap(os.path.join(DATA_ROOT, cam, name), mask, roi,
                             os.path.join(OUT_DIR, f"{cam}_{i+1:02d}_{name}.png"),
                             title=f"score={s:.3f}  rank#{i+1}")

        # 全批次的分布，用于阈值
        print(f"\n  【阈值建议】")
        for k in [4, 6, 8, 10]:
            print(f"    mean+{k}σ = {arr.mean()+k*arr.std():.3f}")

    print(f"\n复核热图已存: {OUT_DIR}")
