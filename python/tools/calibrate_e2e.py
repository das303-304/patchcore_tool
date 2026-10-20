# -*- coding: utf-8 -*-
"""用一体化 ONNX 自身重新标定阈值（全量 OK 图 + 全部 NG 图）。

为什么必须做：e2e 的缩放由 onnxruntime 的 antialias 实现，与 PIL 有 1~3 灰阶差异，
分数会偏移 0.00~0.08。既然生产环境跑的就是 e2e 模型，阈值就必须用 e2e 自己的
分数分布来定，不能沿用 PIL 链路的数字。

输出：models_onnx/<site>/meta_e2e.json
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

CASES = [
    ("皮料相机1", "CAM1"),
    ("皮料相机2", "CAM2"),
]


def score_file(sess, path):
    g = np.asarray(PIL.Image.open(path).convert("L"), dtype=np.uint8)[None, None]
    return float(sess.run(None, {"gray": g})[0].ravel()[0])


def main():
    for site, ng_dir in CASES:
        d = os.path.join(HERE, "models_onnx", site)
        sess = ort.InferenceSession(os.path.join(d, "patchcore_e2e.onnx"),
                                    providers=["CPUExecutionProvider"])
        ok_dir = os.path.join(OK_ROOT, site)
        files = sorted(os.listdir(ok_dir))

        t0 = time.perf_counter()
        scores = []
        for i, f in enumerate(files):
            scores.append(score_file(sess, os.path.join(ok_dir, f)))
            if (i + 1) % 100 == 0:
                print(f"  {site} {i + 1}/{len(files)}", flush=True)
        scores = np.array(scores)
        elapsed = time.perf_counter() - t0

        mean, std = float(scores.mean()), float(scores.std())
        p99 = float(np.percentile(scores, 99))
        mx = float(scores.max())
        amx = int(np.argmax(scores))
        thr6 = mean + 6 * std
        thr = round(max(thr6, mx * 1.02), 2)

        # NG 检出
        ng_path = os.path.join(NG_ROOT, ng_dir)
        ng_files = [f for f in sorted(os.listdir(ng_path)) if f.endswith(".jpg")]
        ng = {}
        for f in ng_files:
            ng[f] = score_file(sess, os.path.join(ng_path, f))
        real_ng = {k: v for k, v in ng.items() if not k.startswith("29_")}
        hit = sum(1 for v in real_ng.values() if v >= thr)

        print()
        print("=" * 84)
        print(f"  [{site}]  e2e 全量标定   {len(files)} 张 OK  ({elapsed / len(files) * 1000:.0f} ms/张)")
        print("=" * 84)
        print(f"  正常分布: 均值 {mean:.4f}  std {std:.4f}  p99 {p99:.4f}  最大 {mx:.4f}")
        print(f"  最大分来自: {files[amx]}")
        print(f"  mean+6σ = {thr6:.4f}   →  阈值 {thr}")
        print(f"  NG 检出: {hit}/{len(real_ng)}"
              f"  分数范围 {min(real_ng.values()):.3f} ~ {max(real_ng.values()):.3f}")
        below = {k: v for k, v in real_ng.items() if v < thr}
        if below:
            print(f"  未检出: {below}")
        print()

        with open(os.path.join(d, "meta_e2e.json"), "w", encoding="utf-8") as f:
            json.dump(dict(
                site=site, n_ok=len(files),
                mean=round(mean, 6), std=round(std, 6),
                p99=round(p99, 6), max=round(mx, 6), max_file=files[amx],
                threshold=thr,
                ng_hit=hit, ng_total=len(real_ng),
                ng_scores={k: round(v, 4) for k, v in real_ng.items()},
                ms_per_image=round(elapsed / len(files) * 1000, 1),
            ), f, ensure_ascii=False, indent=2)
        print(f"  已写入 {os.path.join(d, 'meta_e2e.json')}")


if __name__ == "__main__":
    main()
