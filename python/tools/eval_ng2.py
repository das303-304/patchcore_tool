"""NG 检出评估 —— 正确的定位指标。

前一版指标有缺陷：要求热图峰值精确落在画的线上。
但画的线很细（原图约 10px 高），映射到 224x224 的掩码坐标系只有约 1.8 像素，
而掩码还经过 sigma=4 的高斯平滑 —— 要求峰值命中这么细的线不现实。

改用两个合理指标：
  1. 峰值到最近缺陷像素的距离（掩码坐标系，单位像素）
  2. 峰值是否落在「膨胀后」的缺陷区域内（膨胀 5px，容忍平滑扩散）

同时给出：热图前 1% 区域对缺陷的召回 / 精度。
"""
import os
import sys
import json

import numpy as np
import PIL.Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from infer_custom import LeatherInspector

OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"

SITES = {
    "CAM1": dict(ok="皮料相机1", model=os.path.join(HERE, "models", "cam1"),
                 roi=(540, 30, 1780, 1270), thr=1.05),
    "CAM2": dict(ok="皮料相机2", model=os.path.join(HERE, "models", "cam2"),
                 roi=(15, 30, 1400, 1270), thr=0.85),
}

DIFF_TH = 20
DILATE = 5          # 容忍高斯平滑造成的扩散
RESIZE, IMAGESIZE = 256, 224


def gt_mask_in_model_coords(ok_path, ng_path, roi):
    """缺陷掩码，映射到模型输出（224x224）坐标系。"""
    a = np.asarray(PIL.Image.open(ok_path).convert("L")).astype(np.int16)
    b = np.asarray(PIL.Image.open(ng_path).convert("L")).astype(np.int16)
    if a.shape != b.shape:
        return None
    d = (np.abs(a - b) > DIFF_TH).astype(np.float32)

    x0, y0, x1, y1 = roi
    c = d[y0:y1, x0:x1]
    im = PIL.Image.fromarray((c * 255).astype(np.uint8))
    im = im.resize((RESIZE, RESIZE), PIL.Image.NEAREST)
    w, h = im.size
    left, top = int(round((w - IMAGESIZE) / 2.0)), int(round((h - IMAGESIZE) / 2.0))
    im = im.crop((left, top, left + IMAGESIZE, top + IMAGESIZE))
    return np.asarray(im) > 127


if __name__ == "__main__":
    print("=" * 100)
    print("NG 检出评估（正确指标：峰值到缺陷的距离 + 膨胀命中）")
    print("注意：NG 为用户人工绘制，结论是乐观上界，不等于真实产线表现")
    print("=" * 100)

    summary = {}
    for cam, cfg in SITES.items():
        ng_dir = os.path.join(NG_ROOT, cam)
        files = sorted(f for f in os.listdir(ng_dir) if f.lower().endswith(".jpg"))
        print(f"\n{'='*100}")
        print(f"[{cam}] {len(files)} 张   阈值 {cfg['thr']}")
        print(f"{'='*100}")

        det = LeatherInspector(cfg["model"], roi=cfg["roi"])
        print()

        print(f"  {'文件':<30}{'OK分':>8}{'NG分':>8}{'判定':>7}"
              f"{'峰值距缺陷':>11}{'膨胀命中':>9}{'召回':>8}{'精度':>8}")
        print("  " + "-" * 88)

        rows = []
        for f in files:
            ok_p = os.path.join(OK_ROOT, cfg["ok"], f)
            ng_p = os.path.join(ng_dir, f)
            if not os.path.exists(ok_p):
                continue

            s_ok, _ = det.predict(ok_p)
            s_ng, mask = det.predict(ng_p)
            gt = gt_mask_in_model_coords(ok_p, ng_p, cfg["roi"])
            if gt is None:
                continue

            # 峰值到最近缺陷像素的距离
            dist = ndimage.distance_transform_edt(~gt)
            py, px = np.unravel_index(np.argmax(mask), mask.shape)
            dmin = float(dist[py, px])

            # 膨胀后是否命中
            gt_dil = ndimage.binary_dilation(gt, iterations=DILATE)
            hit_dil = bool(gt_dil[py, px])

            # 前 1% 热区
            th = np.percentile(mask, 99)
            hot = mask >= th
            inter = (hot & gt).sum()
            prec = inter / max(hot.sum(), 1)
            rec = inter / max(gt.sum(), 1)

            verdict = "NG ✓" if s_ng >= cfg["thr"] else "漏报"
            print(f"  {f:<30}{s_ok:8.3f}{s_ng:8.3f}{verdict:>7}"
                  f"{dmin:11.1f}{'是' if hit_dil else '否':>9}"
                  f"{rec*100:7.1f}%{prec*100:7.1f}%")

            rows.append(dict(file=f, ok=s_ok, ng=s_ng, hit=bool(s_ng >= cfg["thr"]),
                             peak_dist=round(dmin, 2), hit_dilated=hit_dil,
                             recall=round(float(rec), 3),
                             precision=round(float(prec), 3)))

        n = len(rows)
        n_hit = sum(r["hit"] for r in rows)
        n_dil = sum(r["hit_dilated"] for r in rows)
        dists = [r["peak_dist"] for r in rows]

        print(f"\n  检出率      : {n_hit}/{n} = {n_hit/max(n,1)*100:.1f}%")
        print(f"  定位命中(膨胀{DILATE}px): {n_dil}/{n} = {n_dil/max(n,1)*100:.1f}%")
        print(f"  峰值距缺陷距离: 中位 {np.median(dists):.1f} px  "
              f"最大 {max(dists):.1f} px  (掩码 224x224)")
        print(f"  平均召回(前1%热区覆盖缺陷): "
              f"{np.mean([r['recall'] for r in rows])*100:.1f}%")
        print(f"  平均精度(前1%热区落在缺陷): "
              f"{np.mean([r['precision'] for r in rows])*100:.1f}%")
        print(f"  OK 原图最高分: {max(r['ok'] for r in rows):.3f}  "
              f"(阈值 {cfg['thr']}，无误报)")

        summary[cam] = dict(n=n, detected=n_hit, dil_hit=n_dil,
                            rate=round(n_hit/max(n,1), 3),
                            thr=cfg["thr"], rows=rows)

    tot = sum(s["n"] for s in summary.values())
    hit = sum(s["detected"] for s in summary.values())
    dil = sum(s["dil_hit"] for s in summary.values())
    print(f"\n{'='*100}")
    print(f"合计: 检出 {hit}/{tot} = {hit/max(tot,1)*100:.1f}%   "
          f"定位命中 {dil}/{tot} = {dil/max(tot,1)*100:.1f}%")
    print(f"{'='*100}")

    out = os.path.join(HERE, "testimgs", "ng_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"明细: {out}")
