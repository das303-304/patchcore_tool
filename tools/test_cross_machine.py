"""验证：图像归一化能否真正消除跨机台的分数偏移。

这是对第二层防线的实证检验，不是纸上谈兵。
做法：拿真实的 123 批次干净图当「一块料」，
      模拟 4 种机台差异（曝光/色温/尺度），
      对比「不归一化直接送模型」vs「归一化后送模型」的分数漂移。

如果归一化有效，后者的跨机台分数离散度应显著小于前者。
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

from crossval_123 import IMGS, SRC, build_bank
from norm_pipeline import Normalizer

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE, IMAGESIZE = 256, 224

# 用 123 批次里那张「干净图」当基准料
CLEAN = IMGS["A_smooth_noline"]
DEFECT = IMGS["D_faint_crease"]     # 有斜折痕的，用来验证缺陷是否还检得出


def make_variants(img):
    """模拟 4 台机器的成像差异。"""
    out = {"基准机": img}

    # 机台B：曝光偏低（环境光暗 / 曝光时间短）
    a = np.asarray(img).astype(np.float32) * 0.72
    out["B 曝光-28%"] = PIL.Image.fromarray(a.clip(0, 255).astype(np.uint8))

    # 机台C：色温偏暖（换了一批灯 or 环境光混入）
    a = np.asarray(img).astype(np.float32)
    a[..., 0] *= 1.10; a[..., 2] *= 0.90
    out["C 色温偏暖"] = PIL.Image.fromarray(a.clip(0, 255).astype(np.uint8))

    # 机台D：对比度低（打光角度不同，明暗对比弱）
    a = np.asarray(img).astype(np.float32)
    a = (a - 128) * 0.68 + 128 + 15
    out["D 对比度-32%"] = PIL.Image.fromarray(a.clip(0, 255).astype(np.uint8))

    return out


def score_one(bank, img, normalizer=None):
    im = normalizer(img) if normalizer else img.convert("RGB")
    tf = T.Compose([T.Resize(RESIZE), T.CenterCrop(IMAGESIZE),
                    T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    s, m = bank.predict(tf(im).unsqueeze(0))
    return float(s[0])


if __name__ == "__main__":
    # 建一个只含这张干净图的记忆库
    clean_p = os.path.join(SRC, CLEAN)
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        bank = build_bank([clean_p], RESIZE, IMAGESIZE, 0.999)
    finally:
        sys.stdout = old

    print("=" * 84)
    print("实验一：同一张「好料」在 4 台机器上的分数（理想情况应全部接近 0）")
    print("=" * 84)

    clean_img = PIL.Image.open(clean_p).convert("RGB")
    variants = make_variants(clean_img)

    # 关掉白平衡/灰度拉伸的归一化器（只做几何，不做光度）
    norm_raw = Normalizer(mm_per_pixel=0.0421, target_mm_per_pixel=0.0421,
                          target_size=(IMAGESIZE, IMAGESIZE), roi_mm=None,
                          gray_stretch=False, white_balance=False)
    # 完整归一化器
    norm_full = Normalizer(mm_per_pixel=0.0421, target_mm_per_pixel=0.0421,
                           target_size=(IMAGESIZE, IMAGESIZE), roi_mm=None,
                           gray_stretch=True, white_balance=True,
                           target_mean=128.0, target_std=40.0)

    print(f"\n{'机台':<16}{'不归一化':>12}{'仅几何':>12}{'完整归一化':>14}")
    print("-" * 84)
    raw_list, full_list = [], []
    for nm, v in variants.items():
        s0 = score_one(bank, v)
        s1 = score_one(bank, v, norm_raw)
        s2 = score_one(bank, v, norm_full)
        raw_list.append(s0); full_list.append(s2)
        print(f"{nm:<16}{s0:12.3f}{s1:12.3f}{s2:14.3f}")

    print("-" * 84)
    print(f"{'跨机台波动(极差)':<16}{max(raw_list)-min(raw_list):12.3f}"
          f"{'':12s}{max(full_list)-min(full_list):14.3f}")
    print(f"{'跨机台标准差':<16}{np.std(raw_list):12.3f}{'':12s}{np.std(full_list):14.3f}")
    print(f"\n结论：归一化把跨机台分数波动从 "
          f"{max(raw_list)-min(raw_list):.3f} 压到 {max(full_list)-min(full_list):.3f}")

    print("\n" + "=" * 84)
    print("实验二：缺陷图在 4 台机器上的分数（理想情况应全部接近缺陷分）")
    print("=" * 84)

    def_img = PIL.Image.open(os.path.join(SRC, DEFECT)).convert("RGB")
    dvars = make_variants(def_img)
    print(f"\n{'机台':<16}{'不归一化':>12}{'完整归一化':>14}{'':>10}")
    print("-" * 84)
    draw, dfull = [], []
    for nm, v in dvars.items():
        s0 = score_one(bank, v)
        s2 = score_one(bank, v, norm_full)
        draw.append(s0); dfull.append(s2)
        print(f"{nm:<16}{s0:12.3f}{s2:14.3f}")
    print("-" * 84)
    print(f"{'跨机台极差':<16}{max(draw)-min(draw):12.3f}"
          f"{max(dfull)-min(dfull):14.3f}")

    print("\n" + "=" * 84)
    print("实验三：归一化后，好料与缺陷是否仍然可分？")
    print("=" * 84)
    print(f"\n  不归一化：好料 {min(raw_list):.3f}~{max(raw_list):.3f}  vs  "
          f"缺陷 {min(draw):.3f}~{max(draw):.3f}")
    print(f"  归一化后：好料 {min(full_list):.3f}~{max(full_list):.3f}  vs  "
          f"缺陷 {min(dfull):.3f}~{max(dfull):.3f}")

    sep_raw = min(draw) - max(raw_list)
    sep_norm = min(dfull) - max(full_list)
    print(f"\n  分离度：  不归一化 {sep_raw:+.3f}    归一化后 {sep_norm:+.3f}")
    print("\n  （分离度 = 缺陷最低分 - 好料最高分，越大越安全）")
