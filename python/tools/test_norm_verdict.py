"""归一化的真实作用边界 —— 做了 3 组实验后的诚实结论。

关键发现：
  - 归一化确实把像素统计拉齐了（mean 127.5, std 31.7，四台机器几乎一致）
  - 但「绝对分数」整体抬高（好料 0.06 -> 6.4），因为对比度增强放大了纹理
  - 跨机台相对波动：不归一化 0.337 vs 归一化 0.258 —— 改善有限

真正要问的不是「分数绝对值」，而是「判定是否一致」。
"""
import os
import sys
import io

import numpy as np
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

from crossval_123 import IMGS, SRC, build_bank
from norm_pipeline import Normalizer
from test_cross_machine import make_variants, score_one, RESIZE, IMAGESIZE

CLEAN = IMGS["A_smooth_noline"]

if __name__ == "__main__":
    clean_p = os.path.join(SRC, IMGS["A_smooth_noline"])
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        bank = build_bank([clean_p], RESIZE, IMAGESIZE, 0.999)
    finally:
        sys.stdout = old

    norm_full = Normalizer(mm_per_pixel=0.0421, target_mm_per_pixel=0.0421,
                           target_size=(IMAGESIZE, IMAGESIZE), roi_mm=None,
                           gray_stretch=True, white_balance=True,
                           target_mean=128.0, target_std=40.0)

    # 用「好料 + 4 张缺陷图」组成一个待检队列
    queue = [("好料", CLEAN, 0)]
    for i, k in enumerate(["B_faint_line", "C_line_topleft",
                           "D_faint_crease", "E_curve_scratch"], 1):
        queue.append((f"缺陷{i}", IMGS[k], 1))

    print("=" * 92)
    print("决定性检验：各机台上「好料 vs 缺陷」的判定是否一致")
    print("=" * 92)

    # 基准机上的分数，用来定基准阈值
    base_scores = []
    for label, fn, is_def in queue:
        img = PIL.Image.open(os.path.join(SRC, fn)).convert("RGB")
        s = score_one(bank, img)
        base_scores.append((label, s, is_def))

    good_s = [s for _, s, d in base_scores if d == 0]
    def_s = [s for _, s, d in base_scores if d == 1]

    for mode_name, use_norm in [("不归一化", False), ("完整归一化", True)]:
        nz = norm_full if use_norm else None
        all_scores = {}

        # 对每台机器，分别用「该机器自己的好料」定阈值（这才是实际部署做法）
        print(f"\n{'-'*92}")
        print(f"【{mode_name}】")
        print(f"{'机台':<14}{'好料分':>9}{'缺陷分区间':>16}{'分离度':>10}{'推荐阈值':>10}{'判定':>8}")
        print("-" * 92)

        for mname in ["基准机", "B 曝光-28%", "C 色温偏暖", "D 对比度-32%"]:
            g, d = [], []
            for label, fn, is_def in queue:
                img = PIL.Image.open(os.path.join(SRC, fn)).convert("RGB")
                v = make_variants(img)[mname]
                s = score_one(bank, v, nz)
                (d if is_def else g).append(s)
            sep = min(d) - max(g)
            thr = (max(g) + min(d)) / 2
            ok = "完美" if sep > 0 else "重叠"
            print(f"{mname:<14}{max(g):9.3f}{min(d):>9.3f}~{max(d):<6.3f}"
                  f"{sep:>10.3f}{thr:>10.3f}{ok:>8}")
