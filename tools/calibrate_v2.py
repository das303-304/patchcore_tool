# -*- coding: utf-8 -*-
"""v2 模型（ROI 由外部裁剪送入）的全量标定。

与 calibrate_e2e.py 的区别：
  v2 的输入是「已裁剪好的 ROI 图」，不再是整幅图，所以这里先按 meta.roi 裁剪。

同时输出「置信度」映射的参考值，供 PLC 侧确认量级：
  conf = clamp(1 - score/阈值, 0, 1) * 1000
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnxruntime as ort
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"

CASES = [("皮料相机1", "cam1"), ("皮料相机2", "cam2")]


def load_roi(site_dir):
    with open(os.path.join(site_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)["roi"]


def prep(path, roi):
    g = np.asarray(PIL.Image.open(path).convert("L"), dtype=np.uint8)
    return g[roi[1]:roi[3], roi[0]:roi[2]][None, None]


def main():
    for site, tag in CASES:
        d = os.path.join(HERE, "models_onnx", site)
        roi = load_roi(d)
        sess = ort.InferenceSession(os.path.join(d, "patchcore_roi.onnx"),
                                    providers=["CPUExecutionProvider"])

        ok_dir = os.path.join(OK_ROOT, site)
        files = sorted(os.listdir(ok_dir))
        t0 = time.perf_counter()
        scores = []
        for i, f in enumerate(files):
            x = prep(os.path.join(ok_dir, f), roi)
            scores.append(float(sess.run(None, {"gray": x})[0].ravel()[0]))
            if (i + 1) % 100 == 0:
                print(f"  {site} {i + 1}/{len(files)}", flush=True)
        scores = np.array(scores)
        elapsed = (time.perf_counter() - t0) / len(files) * 1000

        mean, std = float(scores.mean()), float(scores.std())
        p99, mx = float(np.percentile(scores, 99)), float(scores.max())
        amx = int(np.argmax(scores))
        thr6 = mean + 6 * std
        thr = round(max(thr6, mx * 1.02), 2)

        ng_path = os.path.join(NG_ROOT, {"cam1": "CAM1", "cam2": "CAM2"}[tag])
        ng = {}
        for f in sorted(os.listdir(ng_path)):
            if not f.endswith(".jpg"):
                continue
            ng[f] = float(sess.run(None, {"gray": prep(os.path.join(ng_path, f), roi)})[0].ravel()[0])
        real = {k: v for k, v in ng.items() if not k.startswith("29_")}
        hit = sum(1 for v in real.values() if v >= thr)

        def conf(s):
            return max(0.0, min(1.0, 1.0 - s / thr)) * 1000

        print()
        print("=" * 88)
        print(f"  [{site}] v2 标定   ROI={roi}   {len(files)} 张 OK   {elapsed:.0f} ms/张")
        print("=" * 88)
        print(f"  正常分布: 均值 {mean:.4f}  std {std:.4f}  p99 {p99:.4f}  最大 {mx:.4f} ({files[amx]})")
        print(f"  mean+6σ = {thr6:.4f}  →  阈值 {thr}")
        print(f"  NG 检出: {hit}/{len(real)}   分数 {min(real.values()):.3f} ~ {max(real.values()):.3f}")
        below = {k: v for k, v in real.items() if v < thr}
        if below:
            print(f"  未检出: {below}")
        print()
        print(f"  置信度映射 conf = clamp(1 - score/{thr}, 0, 1) × 1000：")
        print(f"    正常均值 {mean:.3f} → {conf(mean):6.0f}")
        print(f"    正常最大 {mx:.3f} → {conf(mx):6.0f}")
        print(f"    阈值处   {thr:.3f} → {conf(thr):6.0f}")
        print(f"    最弱 NG  {min(real.values()):.3f} → {conf(min(real.values())):6.0f}")
        print()

        out = dict(site=site, model="patchcore_roi.onnx", roi=roi,
                   n_ok=len(files), mean=round(mean, 6), std=round(std, 6),
                   p99=round(p99, 6), max=round(mx, 6), max_file=files[amx],
                   threshold=thr, ng_hit=hit, ng_total=len(real),
                   ng_scores={k: round(v, 4) for k, v in real.items()},
                   ms_per_image=round(elapsed, 1),
                   confidence_at_mean=round(conf(mean), 0),
                   confidence_at_max=round(conf(mx), 0))
        with open(os.path.join(d, "meta_v2.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  已写入 meta_v2.json")


if __name__ == "__main__":
    main()
