"""NG 检出可视化：把画的缺陷轮廓与热图叠在一起，直观对照。"""
import os
import sys

import numpy as np
import PIL.Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from infer_custom import LeatherInspector
from eval_ng2 import gt_mask_in_model_coords, OK_ROOT, NG_ROOT, SITES

OUT = os.path.join(HERE, "testimgs", "ng_vis")
os.makedirs(OUT, exist_ok=True)

# 代表性样本：强改动 / 弱改动但检出 / 漏报
PICKS = [
    ("CAM1", "43_20261008_101441.jpg", "强改动·检出"),
    ("CAM1", "147_20261008_101941.jpg", "强改动·检出"),
    ("CAM1", "75_20261008_101609.jpg", "弱改动·检出"),
    ("CAM1", "16_20261008_101331.jpg", "弱改动·漏报"),
    ("CAM2", "89_20261008_101648.jpg", "强改动·检出"),
    ("CAM2", "134_20261008_101852.jpg", "弱改动·漏报"),
]

if __name__ == "__main__":
    dets = {}
    for cam, cfg in SITES.items():
        dets[cam] = LeatherInspector(cfg["model"], roi=cfg["roi"])

    for cam, f, tag in PICKS:
        cfg = SITES[cam]
        ok_p = os.path.join(OK_ROOT, cfg["ok"], f)
        ng_p = os.path.join(NG_ROOT, cam, f)
        if not os.path.exists(ok_p) or not os.path.exists(ng_p):
            continue

        s_ok, _ = dets[cam].predict(ok_p)
        s_ng, mask = dets[cam].predict(ng_p)
        gt = gt_mask_in_model_coords(ok_p, ng_p, cfg["roi"])

        x0, y0, x1, y1 = cfg["roi"]
        img_ok = PIL.Image.open(ok_p).convert("RGB").crop((x0, y0, x1, y1))
        img_ng = PIL.Image.open(ng_p).convert("RGB").crop((x0, y0, x1, y1))

        # 把 224 的 gt 反向映射回原图尺寸：
        # 正向是 crop(roi) -> Resize(256) -> CenterCrop(224)
        # 反向须先贴回 256 画布（补回被裁掉的边），再放大到 ROI 尺寸
        crop_px = (256 - 224) // 2          # = 16
        canvas = np.zeros((256, 256), dtype=np.uint8)
        canvas[crop_px:crop_px + 224, crop_px:crop_px + 224] = (gt * 255).astype(np.uint8)
        g_big = np.asarray(
            PIL.Image.fromarray(canvas).resize((x1 - x0, y1 - y0), PIL.Image.NEAREST)
        ) > 127

        verdict = "NG ✓" if s_ng >= cfg["thr"] else "漏报 ✗"
        color = "#c62828" if s_ng >= cfg["thr"] else "#ef6c00"

        fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
        axes[0].imshow(img_ok)
        axes[0].set_title(f"OK 原图\nscore={s_ok:.3f}", fontsize=10)
        axes[1].imshow(img_ng)
        axes[1].set_title(f"NG（人工画线）\nscore={s_ng:.3f}  {verdict}",
                          fontsize=10, color=color)
        axes[2].imshow(mask, cmap="jet")
        axes[2].set_title("异常热图", fontsize=10)
        # 叠加：热图 + 缺陷轮廓（青色描边）
        ax = axes[3]
        ax.imshow(img_ng)
        ax.imshow(mask, cmap="jet", alpha=0.45)
        edge = ndimage.binary_dilation(g_big, iterations=3) & ~g_big
        ov = np.zeros((*edge.shape, 4))
        ov[edge] = [0, 1, 1, 1]      # 青色描出画的缺陷位置
        ax.imshow(ov)
        ax.set_title("叠加（青线=画的缺陷）", fontsize=10)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"[{cam}] {f}   {tag}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{cam}_{f}.png"), dpi=92)
        plt.close(fig)
        print(f"  {cam}/{f:<32} OK {s_ok:.3f} -> NG {s_ng:.3f}  {verdict}")

    print(f"\n可视化已存: {OUT}")
