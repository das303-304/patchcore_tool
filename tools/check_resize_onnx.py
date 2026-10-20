# -*- coding: utf-8 -*-
"""对比三种「下采样到 256」的实现，判断能否把缩放搬进 ONNX 图。

为何要这个实验：
  C# 侧写缩放极易与 Python(PIL) 不一致，实测 cv2 INTER_LINEAR 会让正常料
  从 0.48 跳到 1.31（阈值 1.05）→ 直接误报。
  如果把 Resize 放进 ONNX 图，Python 与 C# 都只需喂「ROI 裁剪后的原始像素」，
  缩放由 onnxruntime 统一完成，跨语言天然一致。

对比：
  PIL      : Image.BILINEAR（当前参考，带抗锯齿）
  ONNX-AA  : Resize(opset18, mode=linear, antialias=1, half_pixel)
  ONNX-NA  : Resize(opset18, mode=linear, antialias=0, half_pixel)

输出：缩放后像素最大差、最终异常分数差、判定是否翻转。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnx
import onnxruntime as ort
import PIL.Image
from onnx import TensorProto, helper

HERE = os.path.dirname(os.path.abspath(__file__))
OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"

CASES = [
    ("皮料相机1", "皮料相机1", (540, 30, 1780, 1270), "CAM1", 1.05),
    ("皮料相机2", "皮料相机2", (15, 30, 1400, 1270), "CAM2", 0.85),
]


def build_resize_model(path, antialias):
    """输入 (1,3,H,W) float32 0~255，输出按 sizes 缩放。"""
    node = helper.make_node(
        "Resize", ["X", "", "", "sizes"], ["Y"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="round_prefer_floor",
        antialias=antialias,
    )
    graph = helper.make_graph(
        [node], f"resize_aa{antialias}",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, None, None]),
         helper.make_tensor_value_info("sizes", TensorProto.INT64, [4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, None, None])],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def pil_resize(a, nw, nh):
    """a: HWC uint8"""
    return np.asarray(PIL.Image.fromarray(a).resize((nw, nh), PIL.Image.BILINEAR))


def main():
    from onnx_infer import OnnxPatchCore

    sess = {
        True: build_resize_model(os.path.join(HERE, "testimgs", "_rz_aa.onnx"), 1),
        False: build_resize_model(os.path.join(HERE, "testimgs", "_rz_na.onnx"), 0),
    }

    for site, onnx_site, roi, ng_dir, thr in CASES:
        det = OnnxPatchCore(os.path.join(HERE, "models_onnx", onnx_site), threshold=thr)
        resize, isize = det.resize, det.isize
        x0, y0, x1, y1 = roi

        d_ok = os.path.join(OK_ROOT, site)
        items = [(os.path.join(d_ok, f), "OK") for f in sorted(os.listdir(d_ok))[:2]]
        ngp = os.path.join(NG_ROOT, ng_dir)
        items += [(os.path.join(ngp, f), "NG") for f in
                  sorted(x for x in os.listdir(ngp)
                         if x.endswith(".jpg") and not x.startswith("29_"))[:2]]

        print("=" * 96)
        print(f"  [{site}] ROI={roi}  → 短边 {resize}，中心裁 {isize}")
        print("=" * 96)
        print(f"  {'文件':<24}{'类型':>5}{'PIL分':>8}"
              f"{'ONNX-AA':>10}{'差':>8}{'ONNX-NA':>10}{'差':>8}"
              f"{'像素差AA':>10}{'像素差NA':>10}")
        print("  " + "-" * 92)

        for p, kind in items:
            img = PIL.Image.open(p).convert("RGB")
            crop = np.asarray(img.crop(roi))       # HWC uint8
            h, w = crop.shape[:2]
            if w <= h:
                nw, nh = resize, int(resize * h / w)
            else:
                nh, nw = resize, int(resize * w / h)

            ref = pil_resize(crop, nw, nh).astype(np.float32)

            def score_from(arr):
                """arr: 已缩放的 HWC uint8 → 中心裁 + 归一化 + 推理。"""
                left = int(round((arr.shape[1] - isize) / 2.0))
                top = int(round((arr.shape[0] - isize) / 2.0))
                b = arr[top:top + isize, left:left + isize].astype(np.float32)
                f = (b / 255.0 - det.mean) / det.std
                x = np.ascontiguousarray(f.transpose(2, 0, 1)[None], dtype=np.float32)
                e = det.sess.run(None, {det.in_name: x})[0]
                if e.ndim == 3:
                    e = e[0]
                return float(np.max(det._patch_scores(np.ascontiguousarray(e))))

            s_ref = score_from(ref)

            row = {}
            for aa in (True, False):
                xin = np.ascontiguousarray(
                    crop.transpose(2, 0, 1)[None].astype(np.float32))
                out = sess[aa].run(None, {
                    "X": xin, "sizes": np.array([1, 3, nh, nw], np.int64)})[0]
                out = out[0].transpose(1, 2, 0)                # HWC float 0~255
                px = float(np.abs(out - ref).max())
                u8 = np.clip(np.rint(out), 0, 255).astype(np.uint8)
                row[aa] = (score_from(u8), px)

            print(f"  {os.path.basename(p)[:24]:<24}{kind:>5}{s_ref:8.3f}"
                  f"{row[True][0]:10.3f}{row[True][0] - s_ref:+8.3f}"
                  f"{row[False][0]:10.3f}{row[False][0] - s_ref:+8.3f}"
                  f"{row[True][1]:10.2f}{row[False][1]:10.2f}")
        print()

    for aa in (True, False):
        f = os.path.join(HERE, "testimgs", f"_rz_{'aa' if aa else 'na'}.onnx")
        if os.path.exists(f):
            os.remove(f)

    print("=" * 96)
    print("  阈值 cam1=1.05 / cam2=0.85。若某列 OK 图分数越过阈值，则该缩放方式不可用。")
    print("=" * 96)


if __name__ == "__main__":
    main()
