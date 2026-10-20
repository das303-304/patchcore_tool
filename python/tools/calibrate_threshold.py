"""阈值标定 —— 用「无偏 OK 分数 + NG 分数」联合扫描。

方法论（很重要，两个分布必须用可比的口径）：
  OK 分数  → 2 折交叉验证（每张图由没见过它的库评分）→ 代表"新来一张正常料"的分数
  NG 分数  → 用「全量 OK 建的生产库」评分（NG 图从不在库内）→ 代表"来一张缺陷"的分数

注意两种口径的库大小不同：CV 的库只用一半数据（更小，分数偏高=保守）；
生产库用全部数据（分数更低）。所以 OK 分布是偏保守的上界，安全方向。

输出：阈值扫描表 —— 每个候选阈值下的 误报数 / 检出数，据此选阈值。
"""
import os
import sys
import json

import numpy as np
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from validate_ok_batch import (DATA_ROOT, ROI, build_bank, score_paths,
                               IMG_EXT, RESIZE, IMAGESIZE)
from infer_custom import LeatherInspector

NG_ROOT = "C:/Users/Administrator/Desktop/NG"
SITE_OF = {"皮料相机1": "CAM1", "皮料相机2": "CAM2"}
OUT = os.path.join(HERE, "testimgs", "threshold_calib.json")


def list_ok(cam):
    d = os.path.join(DATA_ROOT, cam)
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if os.path.splitext(f)[1].lower() in IMG_EXT]


def cv_ok_scores(cam, pct=0.1):
    """2 折交叉验证，返回 {文件名: 无偏OK分数}。"""
    ps = list_ok(cam)
    A = ps[0::2]
    B = ps[1::2]
    out = {}
    for build, hold in [(A, B), (B, A)]:
        m = build_bank(build, RESIZE, IMAGESIZE, ROI[cam], pct)
        sc, _ = score_paths(m, hold, RESIZE, IMAGESIZE, ROI[cam])
        for p, s in zip(hold, sc):
            out[os.path.basename(p)] = float(s)
    return out


def ng_scores(cam, model_dir):
    """用生产库（全量 OK）给 NG 图打分。"""
    det = LeatherInspector(model_dir, roi=ROI[cam])
    ngdir = os.path.join(NG_ROOT, SITE_OF[cam])
    out = {}
    if not os.path.isdir(ngdir):
        return out
    for f in sorted(os.listdir(ngdir)):
        if not f.lower().endswith(".jpg"):
            continue
        p = os.path.join(ngdir, f)
        s, _ = det.predict(p)
        out[f] = float(s)
    return out


if __name__ == "__main__":
    result = {}
    for cam in ROI:
        print(f"\n{'='*94}")
        print(f"[{cam}]  阈值标定")
        print(f"{'='*94}")

        print("  计算无偏 OK 分数（2 折交叉验证）…")
        ok = cv_ok_scores(cam)
        oks = np.array(list(ok.values()))
        print(f"    OK  n={len(oks)}  均值 {oks.mean():.3f}  std {oks.std():.3f}  "
              f"p95 {np.percentile(oks,95):.3f}  p99 {np.percentile(oks,99):.3f}  "
              f"最大 {oks.max():.3f}")

        model_dir = os.path.join(HERE, "models", cam.replace("皮料", "m").replace("相机1", "cam1").replace("相机2", "cam2"))
        model_dir = os.path.join(HERE, "models",
                                 "cam1" if cam == "皮料相机1" else "cam2")
        print(f"  计算 NG 分数（生产库 {os.path.basename(model_dir)}）…")
        ng = ng_scores(cam, model_dir)
        ngs = np.array(list(ng.values())) if ng else np.array([])
        if len(ngs):
            print(f"    NG  n={len(ngs)}  最低 {ngs.min():.3f}  "
                  f"最高 {ngs.max():.3f}")
        else:
            print("    NG  无样本")

        # 阈值扫描
        print(f"\n  {'阈值':>7}{'误报':>8}{'误报率':>9}{'检出':>7}{'检出率':>9}"
              f"   判定")
        print("  " + "-" * 66)

        best = None
        for thr in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
            fp = int((oks >= thr).sum())
            fpr = fp / len(oks) * 100
            if len(ngs):
                tp = int((ngs >= thr).sum())
                tpr = tp / len(ngs) * 100
                mark = ""
                if fp == 0 and best is None:
                    best = thr
                    mark = "  <- 零误报下最优"
                print(f"  {thr:7.2f}{fp:8d}{fpr:8.1f}%{tp:7d}{tpr:8.1f}%   {mark}")
            else:
                print(f"  {thr:7.2f}{fp:8d}{fpr:8.1f}%")

        # 逐图列出超过候选阈值的 OK 图
        for cand in ([best] if best else []) + [1.05]:
            over = sorted(((n, s) for n, s in ok.items() if s >= cand),
                          key=lambda x: -x[1])
            if over:
                print(f"\n  超过 {cand:.2f} 的 OK 图（共 {len(over)} 张，即误报来源）:")
                for n, s in over[:8]:
                    print(f"    {n:<34} {s:.3f}")
                break

        result[cam] = dict(
            ok_n=len(oks), ok_mean=round(float(oks.mean()), 4),
            ok_std=round(float(oks.std()), 4),
            ok_p99=round(float(np.percentile(oks, 99)), 4),
            ok_max=round(float(oks.max()), 4),
            ok_max_file=max(ok.items(), key=lambda x: x[1])[0],
            ng_n=len(ngs),
            ng_min=round(float(ngs.min()), 4) if len(ngs) else None,
            ng_max=round(float(ngs.max()), 4) if len(ngs) else None,
            suggested_thr=best,
        )

    print(f"\n{'='*94}")
    print("汇总")
    print(f"{'='*94}")
    for cam, r in result.items():
        print(f"  {cam}:")
        print(f"    OK  n={r['ok_n']}  最大 {r['ok_max']} ({r['ok_max_file']})")
        print(f"    NG  n={r['ng_n']}  最低 {r['ng_min']}")
        print(f"    → 建议阈值 {r['suggested_thr']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已存: {OUT}")
