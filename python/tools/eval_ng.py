"""用 NG 图评估检出能力。

重要前提（必须先说清）：
  这批 NG 是**人工在 OK 图上画线**模拟的（用户手工绘制），
  不是真实产线缺陷。所以测出来的"检出率"是**乐观上界**：
    - 画的线对比度高（纯黑/纯白 vs ~130 灰底），比真实划痕好检
    - 没有真实缺陷的形态多样性（压痕、污渍、色差、毛孔异常等）
  结论只能用于**验证阈值是否合理、管线是否通畅**，不能当验收依据。

评估内容：
  1. 图像级：NG 图能否超过阈值报警（检出率）
  2. 图像级：对应的 OK 原图是否保持低分（误报率）
  3. 像素级：热图是否落在画的线上（定位准确度）

用法:
    python eval_ng.py
"""
import os
import sys
import json

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

from validate_ok_batch import MEAN, STD, DEV
from infer_custom import LeatherInspector

OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"

# 相机目录名映射 + 生产模型 + 阈值
SITES = {
    "CAM1": dict(ok="皮料相机1", model=os.path.join(HERE, "models", "cam1"),
                 roi=(540, 30, 1780, 1270), thr=0.91),
    "CAM2": dict(ok="皮料相机2", model=os.path.join(HERE, "models", "cam2"),
                 roi=(15, 30, 1400, 1270), thr=0.88),
}

DIFF_TH = 20          # 判定"改动过"的像素阈值
RESIZE, IMAGESIZE = 256, 224


def diff_mask(ok_path, ng_path, size):
    """用 OK 与 NG 的像素差生成缺陷掩码（原图分辨率）。"""
    a = np.asarray(PIL.Image.open(ok_path).convert("L")).astype(np.int16)
    b = np.asarray(PIL.Image.open(ng_path).convert("L")).astype(np.int16)
    if a.shape != b.shape:
        return None
    return (np.abs(a - b) > DIFF_TH).astype(np.float32)


def score_pair(det, ng_path, roi, resize=RESIZE, imagesize=IMAGESIZE):
    img = PIL.Image.open(ng_path).convert("RGB")
    crop = img.crop(roi)
    t = det.transform(img).unsqueeze(0)
    s, mask = det.model.predict(t)
    return float(s[0]), mask[0], crop


def mask_to_roi(m, roi, target=IMAGESIZE, resize=RESIZE):
    """把原图分辨率掩码映射到模型输出的掩码坐标系。

    必须完整复现推理的几何变换链，否则对不齐：
        crop(roi)  ->  Resize(resize)  ->  CenterCrop(target)
    注意 CenterCrop 会切掉外围，不能直接把 ROI resize 到 target。
    """
    x0, y0, x1, y1 = roi
    c = m[y0:y1, x0:x1]
    im = PIL.Image.fromarray((c * 255).astype(np.uint8))
    im = im.resize((resize, resize), PIL.Image.NEAREST)   # 与 T.Resize 一致

    # 复现 CenterCrop：居中裁 target x target
    w, h = im.size
    left = int(round((w - target) / 2.0))
    top = int(round((h - target) / 2.0))
    im = im.crop((left, top, left + target, top + target))

    return np.asarray(im) > 127


if __name__ == "__main__":
    print("=" * 96)
    print("NG 检出能力评估（注：NG 为人工绘制，结论是乐观上界）")
    print("=" * 96)

    summary = {}
    all_rows = []

    for cam, cfg in SITES.items():
        ng_dir = os.path.join(NG_ROOT, cam)
        files = sorted(f for f in os.listdir(ng_dir) if f.lower().endswith(".jpg"))
        print(f"\n{'='*96}")
        print(f"[{cam}] {len(files)} 张 NG   模型 {os.path.basename(cfg['model'])}"
              f"   阈值 {cfg['thr']}")
        print(f"{'='*96}")

        det = LeatherInspector(cfg["model"], roi=cfg["roi"])
        print()

        print(f"  {'文件':<30}{'OK分':>8}{'NG分':>8}{'判定':>7}"
              f"{'峰值位置命中':>14}")
        print("  " + "-" * 74)

        n_hit = 0
        rows = []
        for f in files:
            ok_p = os.path.join(OK_ROOT, cfg["ok"], f)
            ng_p = os.path.join(ng_dir, f)
            if not os.path.exists(ok_p):
                print(f"  {f:<30}  (OK 批次无同名文件，跳过)")
                continue

            s_ok, _, _ = score_pair(det, ok_p, cfg["roi"])
            s_ng, m_ng, _ = score_pair(det, ng_p, cfg["roi"])
            hit = s_ng >= cfg["thr"]
            n_hit += hit

            # 像素级：热图峰值是否落在画的线上
            gt = diff_mask(ok_p, ng_p, None)
            gt_r = mask_to_roi(gt, cfg["roi"])
            peak = np.unravel_index(np.argmax(m_ng), m_ng.shape)
            in_defect = bool(gt_r[peak])
            # 热图 top1% 区域与缺陷区域的重叠度
            th = np.percentile(m_ng, 99)
            hot = m_ng >= th
            inter = (hot & gt_r).sum()
            prec = inter / max(hot.sum(), 1)
            rec = inter / max(gt_r.sum(), 1)

            print(f"  {f:<30}{s_ok:8.3f}{s_ng:8.3f}"
                  f"{'NG ✓' if hit else '漏报':>7}{'命中' if in_defect else '偏离':>14}")
            rows.append(dict(file=f, ok=s_ok, ng=s_ng, hit=bool(hit),
                             peak_in_defect=in_defect,
                             heat_precision=round(float(prec), 3),
                             heat_recall=round(float(rec), 3),
                             n_defect_px=int(gt.sum())))

        n = len(rows)
        print(f"\n  检出率: {n_hit}/{n} = {n_hit/max(n,1)*100:.1f}%")
        if rows:
            print(f"  NG 分数范围: {min(r['ng'] for r in rows):.3f} ~ "
                  f"{max(r['ng'] for r in rows):.3f}")
            print(f"  OK 分数范围: {min(r['ok'] for r in rows):.3f} ~ "
                  f"{max(r['ok'] for r in rows):.3f}")
            n_pk = sum(r["peak_in_defect"] for r in rows)
            print(f"  热图峰值落在缺陷上: {n_pk}/{n} = {n_pk/max(n,1)*100:.1f}%")
            print(f"  平均热图精度(前1%区域命中缺陷比例): "
                  f"{np.mean([r['heat_precision'] for r in rows])*100:.1f}%")
            print(f"  平均热图召回(缺陷被前1%区域覆盖比例): "
                  f"{np.mean([r['heat_recall'] for r in rows])*100:.1f}%")

        summary[cam] = dict(n=n, detected=n_hit,
                            rate=round(n_hit / max(n, 1), 3),
                            ng_min=round(min((r['ng'] for r in rows), default=0), 3),
                            ng_max=round(max((r['ng'] for r in rows), default=0), 3),
                            ok_max=round(max((r['ok'] for r in rows), default=0), 3),
                            thr=cfg["thr"], rows=rows)
        all_rows.extend(rows)

    print(f"\n{'='*96}")
    print("汇总")
    print(f"{'='*96}")
    for cam, s in summary.items():
        print(f"  {cam}: {s['detected']}/{s['n']} 检出 ({s['rate']*100:.1f}%)  "
              f"NG 分 {s['ng_min']:.3f}~{s['ng_max']:.3f}  "
              f"OK 分最高 {s['ok_max']:.3f}  阈值 {s['thr']}")

    tot = len(all_rows)
    hit = sum(r["hit"] for r in all_rows)
    print(f"\n  合计: {hit}/{tot} = {hit/max(tot,1)*100:.1f}%")
    fp = sum(1 for r in all_rows if r["ok"] >= r.get("thr", 0.9))
    print(f"  对应 OK 原图误报: {fp}/{tot}")

    out = os.path.join(HERE, "testimgs", "ng_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  明细已存: {out}")
