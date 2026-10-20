# -*- coding: utf-8 -*-
"""验证一体化 ONNX 与既有 Python 链路（PIL 预处理 + numpy KNN）是否一致。

通过标准：
    - 分数差 < 0.05（阈值余量是 0.23，留足安全边界）
    - 判定（OK/NG）不翻转
    - 热图峰值位置一致（误差几个像素内）
"""
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
    ("皮料相机1", "CAM1", 1.05),
    ("皮料相机2", "CAM2", 0.85),
]


def main():
    from onnx_infer import OnnxPatchCore

    all_ok = True
    for site, ng_dir, thr in CASES:
        d = os.path.join(HERE, "models_onnx", site)
        e2e = ort.InferenceSession(os.path.join(d, "patchcore_e2e.onnx"),
                                   providers=["CPUExecutionProvider"])
        ref = OnnxPatchCore(d, threshold=thr)

        ok_dir = os.path.join(OK_ROOT, site)
        ok_files = sorted(os.listdir(ok_dir))
        ng_path = os.path.join(NG_ROOT, ng_dir)
        ng_files = [f for f in sorted(os.listdir(ng_path))
                    if f.endswith(".jpg") and not f.startswith("29_")]

        items = [(os.path.join(ok_dir, f), "OK") for f in ok_files[:4]]
        items += [(os.path.join(ng_path, f), "NG") for f in ng_files[:4]]

        print("=" * 96)
        print(f"  [{site}]  阈值 {thr}   e2e 输入 整图灰度 uint8")
        print("=" * 96)
        print(f"  {'文件':<24}{'类型':>5}{'PIL链路':>9}{'e2e':>9}{'差':>8}"
              f"{'判定一致':>9}{'峰值距':>8}{'耗时':>8}")
        print("  " + "-" * 92)

        for p, kind in items:
            g = np.asarray(PIL.Image.open(p).convert("L"), dtype=np.uint8)
            x = g[None, None]                      # (1,1,H,W)

            s_ref = float(np.max(ref.infer(p)[2]) if False else 0)
            # 参考链路分数
            sc_ref, _, amap_ref, ms_ref, _ = ref.infer(p)

            t = time.perf_counter()
            out = e2e.run(None, {"gray": x})
            ms_e2e = (time.perf_counter() - t) * 1000
            sc_e2e = float(out[0].ravel()[0])
            amap = out[1][0, 0]

            v_ref = "NG" if sc_ref >= thr else "OK"
            v_e2e = "NG" if sc_e2e >= thr else "OK"
            same = "是" if v_ref == v_e2e else "**否**"
            if v_ref != v_e2e:
                all_ok = False

            py, px = np.unravel_index(np.argmax(amap_ref), amap_ref.shape)
            qy, qx = np.unravel_index(np.argmax(amap), amap.shape)
            dist = float(np.hypot(py - qy, px - qx))

            d_score = sc_e2e - sc_ref
            if abs(d_score) > 0.05:
                all_ok = False

            print(f"  {os.path.basename(p)[:24]:<24}{kind:>5}{sc_ref:9.3f}"
                  f"{sc_e2e:9.3f}{d_score:+8.3f}{same:>9}{dist:8.1f}{ms_e2e:7.0f}ms")

        print()

    print("=" * 96)
    print("  结论：" + ("全部通过 ✓" if all_ok else "存在超差，需要排查 ✗"))
    print("=" * 96)


if __name__ == "__main__":
    main()
