"""检验 123 批次的高分究竟来自哪里。

关键假设：这些是"手拍裁剪"图，裁剪边缘可能引入伪影。
做法：对每张图只用中心 70% 区域重新打分，看分数是否仍成立。
      同时输出热图供人眼复核。
"""
import os
import sys
import io

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

from crossval_123 import IMGS, SRC, build_bank, DEV

OUT = os.path.join(HERE, "testimgs", "hm_123")
os.makedirs(OUT, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE, IMAGESIZE = 256, 224


def center_crop_frac(img, frac):
    w, h = img.size
    dw, dh = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return img.crop((dw, dh, w - dw, h - dh))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank_path = os.path.join(SRC, IMGS["A_smooth_noline"])
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        bank = build_bank([bank_path], RESIZE, IMAGESIZE, 0.999)
    finally:
        sys.stdout = old

    tf = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMAGESIZE),
                    T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    order = ["080d03b9559bf43454d70e0ffb49cc7f.jpg",
             "2f90ec8d3923f40b6c72a12f3e5edfed.jpg",
             "5e87f775ce21de409cea16ea974ed6ea.jpg",
             "0fd8044ee58abaaada8f5db5e183deba(1).jpg",
             "5891e1f45ef89a7119c7469f510aab09(1).jpg"]

    print(f"{'图片':<44}{'全图':>8}{'中心70%':>10}{'变化':>9}")
    print("-" * 74)
    for name in order:
        p = os.path.join(SRC, name)
        img = PIL.Image.open(p).convert("RGB")

        s_full, m_full = bank.predict(tf(img).unsqueeze(0))
        s_full = float(s_full[0]); m_full = m_full[0]

        cropped = center_crop_frac(img, 0.70)
        s_c, m_c = bank.predict(tf(cropped).unsqueeze(0))
        s_c = float(s_c[0]); m_c = m_c[0]

        print(f"{name:<44}{s_full:8.3f}{s_c:10.3f}{s_c-s_full:+9.3f}")

        fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
        axes[0][0].imshow(img); axes[0][0].set_title(
            f"original (full)\nscore={s_full:.2f}", fontsize=9)
        axes[0][1].imshow(m_full, cmap="jet"); axes[0][1].set_title("anomaly map", fontsize=9)
        axes[0][2].imshow(img.resize(m_full.shape[::-1]))
        axes[0][2].imshow(m_full, cmap="jet", alpha=0.55)
        axes[0][2].set_title("overlay", fontsize=9)

        axes[1][0].imshow(cropped); axes[1][0].set_title(
            f"center 70% crop\nscore={s_c:.2f}", fontsize=9)
        axes[1][1].imshow(m_c, cmap="jet"); axes[1][1].set_title("anomaly map", fontsize=9)
        axes[1][2].imshow(cropped.resize(m_c.shape[::-1]))
        axes[1][2].imshow(m_c, cmap="jet", alpha=0.55)
        axes[1][2].set_title("overlay", fontsize=9)
        for r in axes:
            for a in r:
                a.axis("off")
        fig.suptitle(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=92)
        plt.close(fig)

    print(f"\n热图已存: {OUT}")
