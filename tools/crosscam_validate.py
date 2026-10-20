"""跨相机验证 —— 直接回答「每台机器成像不同，库能不能通用」。

三组对照：
  1. 相机1 的库  -> 评相机1 的图   （同域基线）
  2. 相机1 的库  -> 评相机2 的图   （跨相机，测通用性）
  3. 相机2 的库  -> 评相机1 的图   （反向跨相机）

如果跨相机分数与同域分数接近 -> 说明 ROI 归一化后两机可共享一个库
如果跨相机分数显著升高        -> 说明必须每台机器独立建库
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from validate_ok_batch import (DATA_ROOT, ROI, IMG_EXT, build_bank,
                               score_paths, DEV)

RESIZE, IMAGESIZE, PCT = 256, 224, 0.1


def load_cam(cam):
    d = os.path.join(DATA_ROOT, cam)
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if os.path.splitext(f)[1].lower() in IMG_EXT]


def stats(name, s):
    a = np.array(s)
    print(f"  {name:<34} n={len(a):4d}  均值 {a.mean():.3f}  "
          f"std {a.std():.3f}  p95 {np.percentile(a,95):.3f}  最大 {a.max():.3f}")
    return a


if __name__ == "__main__":
    c1 = load_cam("皮料相机1")
    c2 = load_cam("皮料相机2")

    print("=" * 92)
    print("跨相机验证：库能否通用")
    print("=" * 92)

    print(f"\n相机1: {len(c1)} 张   相机2: {len(c2)} 张")
    print(f"ROI 相机1: {ROI['皮料相机1']}")
    print(f"ROI 相机2: {ROI['皮料相机2']}")

    results = {}

    # --- 库1 ---
    print(f"\n{'-'*92}")
    print("【库1】用相机1 全部 OK 图建库")
    print("-" * 92)
    m1 = build_bank(c1, RESIZE, IMAGESIZE, ROI["皮料相机1"], PCT)
    print(f"  记忆库向量数: {m1.anomaly_scorer.nn_method.search_index.ntotal}")
    s1_same, _ = score_paths(m1, c1, RESIZE, IMAGESIZE, ROI["皮料相机1"])
    s1_cross, _ = score_paths(m1, c2, RESIZE, IMAGESIZE, ROI["皮料相机2"])
    results["库1->相机1"] = stats("相机1 的图（同域）", s1_same)
    results["库1->相机2"] = stats("相机2 的图（跨域）", s1_cross)

    # --- 库2 ---
    print(f"\n{'-'*92}")
    print("【库2】用相机2 全部 OK 图建库")
    print("-" * 92)
    m2 = build_bank(c2, RESIZE, IMAGESIZE, ROI["皮料相机2"], PCT)
    print(f"  记忆库向量数: {m2.anomaly_scorer.nn_method.search_index.ntotal}")
    s2_cross, _ = score_paths(m2, c1, RESIZE, IMAGESIZE, ROI["皮料相机1"])
    s2_same, _ = score_paths(m2, c2, RESIZE, IMAGESIZE, ROI["皮料相机2"])
    results["库2->相机2"] = stats("相机2 的图（同域）", s2_same)
    results["库2->相机1"] = stats("相机1 的图（跨域）", s2_cross)

    # --- 判定 ---
    print(f"\n{'='*92}")
    print("结论")
    print(f"{'='*92}")
    same_mx = max(results["库1->相机1"].max(), results["库2->相机2"].max())
    cross_mx = max(results["库1->相机2"].max(), results["库2->相机1"].max())
    same_mean = np.mean([results["库1->相机1"].mean(),
                         results["库2->相机2"].mean()])
    cross_mean = np.mean([results["库1->相机2"].mean(),
                          results["库2->相机1"].mean()])
    print(f"\n  同域：均值 {same_mean:.3f}  最大 {same_mx:.3f}")
    print(f"  跨域：均值 {cross_mean:.3f}  最大 {cross_mx:.3f}")
    print(f"  偏移：均值 {cross_mean-same_mean:+.3f}   "
          f"最大 {cross_mx-same_mx:+.3f}")
    ratio = cross_mx / same_mx if same_mx else float("nan")
    print(f"  最大分数放大倍数: {ratio:.2f}x")

    if ratio < 1.3:
        print("\n  -> 两机可共享一个记忆库（打光/取景已足够一致）")
    elif ratio < 2.0:
        print("\n  -> 勉强可用但建议各自独立库，并各自标定阈值")
    else:
        print("\n  -> 必须每台机器独立建库（成像差异过大）")
