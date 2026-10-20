# -*- coding: utf-8 -*-
"""用新的骨干模板重拼某相机的 patchcore_roi.onnx，记忆库保持不动。

为什么不需要重建库：新骨干与原骨干的特征数值等价
（PyTorch 层 max|diff| = 8.94e-08，ONNX 层 3.94e-06），
记忆库存的就是这些特征向量，因此 memory_bank_v2.npy 可以直接复用。

前置：先跑 _archive/export_fast_base.py 生成新的 models_onnx/_base/model.onnx

用法:
    python _archive/repack_cam.py --cam 1
    python _archive/repack_cam.py --cam 2
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from build_tool import (BASE_MODEL, OUT_ROOT, build_final, build_front)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    cam_dir = os.path.join(OUT_ROOT, "cam%d" % args.cam)
    bank_path = os.path.join(cam_dir, "memory_bank_v2.npy")
    model_path = os.path.join(cam_dir, "patchcore_roi.onnx")
    front_path = os.path.join(cam_dir, "front.onnx")

    print("=" * 72)
    print("  重拼 cam%d" % args.cam)
    print("=" * 72)

    for p, what in [(BASE_MODEL, "骨干模板"), (bank_path, "记忆库")]:
        if not os.path.exists(p):
            raise SystemExit("%s不存在: %s" % (what, p))
    print("  骨干模板: %.1f MB" % (os.path.getsize(BASE_MODEL) / 1048576))
    print("  记忆库  : %.1f MB" % (os.path.getsize(bank_path) / 1048576))

    bank = np.load(bank_path)
    print("  记忆库形状: %s  (不做任何修改)" % (bank.shape,))

    # 旧模型留一份（如果之前没备份）
    if os.path.exists(model_path) and not args.no_backup:
        bak = model_path + ".before-repack"
        if not os.path.exists(bak):
            shutil.copy2(model_path, bak)
            print("  旧模型已备份 -> %s" % os.path.basename(bak))

    t0 = time.time()
    print("\n  [1/2] 生成 front.onnx ...")
    build_front(BASE_MODEL, front_path)
    print("        %.1f MB  用时 %.1fs"
          % (os.path.getsize(front_path) / 1048576, time.time() - t0))

    t1 = time.time()
    print("  [2/2] 拼装 patchcore_roi.onnx ...")
    build_final(front_path, bank, model_path)
    print("        %.1f MB  用时 %.1fs"
          % (os.path.getsize(model_path) / 1048576, time.time() - t1))

    # 冒烟：能被 ORT 加载且输出正确
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()[0]
    outs = [o.name for o in sess.get_outputs()]
    print("\n  输入: %s %s" % (ins.name, ins.shape))
    print("  输出: %s" % outs)
    x = np.random.randint(0, 255, (1, 1, 640, 640)).astype(np.uint8)
    sc, mp = sess.run(["score", "anomaly_map"], {"gray": x})
    print("  score=%s  anomaly_map=%s" % (sc.ravel(), mp.shape))

    print("\n完成。下一步跑 _archive/verify_scores.py --cam %d 做分数比对" % args.cam)


if __name__ == "__main__":
    main()
