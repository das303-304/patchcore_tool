"""用实验 D 的记忆库出热图，并做逐图放大复核。"""
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

from crossval_test_folder import IMGS, SRC, build_bank, DEV

OUT = os.path.join(HERE, "testimgs", "hm_selfbuilt")
os.makedirs(OUT, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE, IMAGESIZE = 256, 224

TRAIN = ["A1_flatsheet_shelf", "F1_plain_low", "B2_deepgroove", "C1_dimple_sheet"]

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [os.path.join(SRC, IMGS[k]) for k in TRAIN]
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        bank = build_bank(paths, RESIZE, IMAGESIZE, 0.999)
    finally:
        sys.stdout = old

    tf = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMAGESIZE),
                    T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    order = ["0fd8044ee58abaaada8f5db5e183deba.jpg",
             "5891e1f45ef89a7119c7469f510aab09.jpg",
             "30868a8726062ae8026762210f6144f2.jpg",
             "2f90ec8d3923f40b6c72a12f3e5edfed(1).jpg",
             "080d03b9559bf43454d70e0ffb49cc7f(1).jpg",
             "5e87f775ce21de409cea16ea974ed6ea(1).jpg",
             "2f90ec8d3923f40b6c72a12f3e5edfed.jpg",
             "1e96d5bec9d09f66cbba087733fa8e76.jpg",
             "080d03b9559bf43454d70e0ffb49cc7f.jpg",
             "5e87f775ce21de409cea16ea974ed6ea.jpg"]

    print(f"{'图片':<46}{'score':>9}{'最高响应占比':>14}")
    for name in order:
        p = os.path.join(SRC, name)
        img = PIL.Image.open(p).convert("RGB")
        t = tf(img).unsqueeze(0)
        s, mask = bank.predict(t)
        m = mask[0]
        # 响应集中度：top1% 像素贡献的能量占比
        flat = np.sort(m.ravel())
        top = flat[int(len(flat) * 0.99):]
        conc = float(top.sum() / max(flat.sum(), 1e-9))
        print(f"{name:<46}{float(s[0]):9.3f}{conc*100:13.1f}%")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].imshow(img); axes[0].set_title(f"original\n{name[:26]}", fontsize=8)
        axes[1].imshow(m, cmap="jet"); axes[1].set_title("anomaly map", fontsize=9)
        axes[2].imshow(img.resize(m.shape[::-1]))
        axes[2].imshow(m, cmap="jet", alpha=0.55)
        axes[2].set_title(f"overlay  score={float(s[0]):.2f}", fontsize=9)
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=95)
        plt.close(fig)
    print(f"\n热图已存: {OUT}")
