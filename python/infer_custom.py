"""用自己的皮料做无训练缺陷检测 —— 推理脚本。

不用 MVTec 数据集，不依赖任何标注。加载自建记忆库，对单张/批量图片出结果。

用法:
    # 单张图片
    python infer_custom.py --model my_models/leather --image test.png

    # 整个文件夹，带阈值判定 + 保存分割热图
    python infer_custom.py --model my_models/leather --dir ./待检图 --threshold 1.67 --save-heatmap

    # 只输出分数 CSV（对接产线 MES / PLC 用）
    python infer_custom.py --model my_models/leather --dir ./待检图 --csv out.csv

输出:
    score  : 图像级异常分数（越大越异常）
    mask   : 像素级缺陷位置，224x224 浮点图（已上采样回 imagesize）

对接产线时提取的判定逻辑（score 阈值来自建库阶段实测）:
    score >= threshold  ->  NG
    score <  threshold  ->  OK
"""
import argparse
import csv
import os
import sys

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "patchcore-inspection")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))

import patchcore.common  # noqa: E402
from patchcore import patchcore as pc  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class LeatherInspector:
    """皮料缺陷检测器。一次加载，反复调用。"""

    def __init__(self, model_dir, device=None, resize=256, imagesize=224, roi=None):
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.imagesize = imagesize
        self.roi = tuple(roi) if roi else None

        # 记忆库目录同时含 .faiss 与 patchcore_params.pkl
        self.model = pc.PatchCore(self.device)
        self.model.load_from_path(
            load_path=model_dir,
            device=self.device,
            nn_method=patchcore.common.FaissNN(False, 4),  # CPU faiss
        )
        steps = []
        if self.roi:
            # ROI 必须在 Resize 之前、原图坐标系下裁 ——
            # 必须与建库时用完全相同的 ROI，否则分数不可比
            steps.append(T.Lambda(lambda im: im.crop(self.roi)))
        steps += [
            T.Resize(resize),
            T.CenterCrop(imagesize),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform = T.Compose(steps)
        print(f"已加载记忆库: {model_dir}")
        print(f"  记忆库规模: {self.model.anomaly_scorer.nn_method.search_index.ntotal}")
        print(f"  设备: {self.device}")
        if self.roi:
            print(f"  ROI: {self.roi}")

    def predict(self, image_path):
        """返回 (score, mask)。"""
        img = PIL.Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        scores, masks = self.model.predict(tensor)
        return float(scores[0]), masks[0]

    def predict_array(self, rgb_array):
        """直接吃 numpy RGB 数组（对接相机 SDK 用）。返回 (score, mask)。"""
        img = PIL.Image.fromarray(rgb_array)
        tensor = self.transform(img).unsqueeze(0)
        scores, masks = self.model.predict(tensor)
        return float(scores[0]), masks[0]


def save_heatmap(image_path, mask, out_path, roi=None):
    """把分割图叠加到原图上，方便人眼复核。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = PIL.Image.open(image_path).convert("RGB")
    if roi:
        # 热图是在 ROI 上算的，叠加图也必须裁到同一区域才能对齐
        img = img.crop(roi)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[1].imshow(mask, cmap="jet")
    axes[1].set_title("anomaly map")
    axes[2].imshow(img.resize(mask.shape[::-1]))
    axes[2].imshow(mask, cmap="jet", alpha=0.5)
    axes[2].set_title("overlay")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="记忆库目录")
    ap.add_argument("--image", help="单张图片")
    ap.add_argument("--dir", help="图片文件夹")
    ap.add_argument("--threshold", type=float, default=None,
                    help="判定阈值；不传则只输出分数")
    ap.add_argument("--csv", help="把结果写入 CSV")
    ap.add_argument("--save-heatmap", action="store_true",
                    help="保存分割热图（存到 <dir>/heatmaps）")
    ap.add_argument("--roi", default=None,
                    help="与建库时相同的 ROI，格式 x0,y0,x1,y1。"
                         "不传则使用整图（可能与建库不一致导致误报）")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("必须指定 --image 或 --dir")

    roi = None
    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
        if len(roi) != 4:
            ap.error("--roi 需要 4 个整数: x0,y0,x1,y1")

    det = LeatherInspector(args.model, roi=roi)

    # 收集待检图片
    if args.image:
        files = [args.image]
    else:
        files = sorted(
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if os.path.splitext(f)[1].lower() in IMG_EXT
        )
        if not files:
            print(f"未在 {args.dir} 找到图片")
            return

    print(f"\n待检 {len(files)} 张\n")
    hd = os.path.join(args.dir, "heatmaps") if args.save_heatmap and args.dir else None
    if hd:
        os.makedirs(hd, exist_ok=True)

    rows = []
    n_ng = 0
    for f in files:
        score, mask = det.predict(f)
        verdict = ""
        if args.threshold is not None:
            verdict = "NG" if score >= args.threshold else "OK"
            if verdict == "NG":
                n_ng += 1
        rows.append((os.path.basename(f), round(score, 4), verdict))
        print(f"  {os.path.basename(f):<40} score={score:8.3f}  {verdict}")
        if hd:
            save_heatmap(f, mask, os.path.join(hd, os.path.basename(f) + ".heat.png"),
                         roi=roi)

    if args.threshold is not None:
        print(f"\n汇总: {len(files)} 张, NG {n_ng}, OK {len(files)-n_ng}, "
              f"阈值 {args.threshold}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            w.writerow(["filename", "score", "verdict"])
            w.writerows(rows)
        print(f"已写入 {args.csv}")


if __name__ == "__main__":
    main()
