# -*- coding: utf-8 -*-
"""检验 C# 侧若用 OpenCvSharp 缩放，结果是否还能对上。

参考实现用的是 PIL.Image.BILINEAR —— PIL 在**缩小时**会按比例放大滤波支撑域，
等价于带抗锯齿的面积加权；而 OpenCV 的 INTER_LINEAR 只取 2×2 邻域，不抗锯齿。
1240 -> 256 是 4.8 倍下采样，两者差异可能不小。

对比三种：
  ref   : PIL BILINEAR（当前 Python / 已验证）
  cv2-L : OpenCV INTER_LINEAR
  cv2-A : OpenCV INTER_AREA（下采样时最接近抗锯齿的行为）

输出：输入张量最大差、patch 分数最大差、图像分数差、判定是否翻转。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"

CASES = [
    ("皮料相机1", os.path.join(HERE, "models_onnx", "皮料相机1"),
     (540, 30, 1780, 1270), "CAM1", 1.05),
    ("皮料相机2", os.path.join(HERE, "models_onnx", "皮料相机2"),
     (15, 30, 1400, 1270), "CAM2", 0.85),
]


def prep_cv2(img, roi, resize, isize, mean, std, flag):
    """与 onnx_infer.preprocess 完全相同的取整规则，只把缩放换成 cv2。"""
    import cv2

    a = np.asarray(img.crop(roi))              # RGB uint8
    h, w = a.shape[:2]
    if w <= h:
        nw, nh = resize, int(resize * h / w)   # 长边 int() 截断，对齐 torchvision
    else:
        nh, nw = resize, int(resize * w / h)
    b = cv2.resize(a, (nw, nh), interpolation=flag)
    left = int(round((nw - isize) / 2.0))
    top = int(round((nh - isize) / 2.0))
    b = b[top:top + isize, left:left + isize]
    f = b.astype(np.float32) / 255.0
    f = (f - mean) / std
    return np.ascontiguousarray(f.transpose(2, 0, 1)[None], dtype=np.float32)


def main():
    import cv2
    from onnx_infer import OnnxPatchCore

    print(f"OpenCV {cv2.__version__}   PIL {PIL.__version__}")
    print()

    for site, onnx_dir, roi, ng_dir, thr in CASES:
        det = OnnxPatchCore(onnx_dir, threshold=thr)
        d_ok = os.path.join(OK_ROOT, site)
        ok_files = sorted(os.listdir(d_ok))[:2]
        ng_path = os.path.join(NG_ROOT, ng_dir)
        ng_files = sorted(f for f in os.listdir(ng_path)
                          if f.endswith(".jpg") and not f.startswith("29_"))[:2]

        items = [(os.path.join(d_ok, f), "OK") for f in ok_files]
        items += [(os.path.join(ng_path, f), "NG") for f in ng_files]

        print("=" * 92)
        print(f"  [{site}]  ROI={roi}")
        print("=" * 92)
        print(f"  {'文件':<26}{'类型':>5}{'ref分数':>9}"
              f"{'cv2-L差':>9}{'cv2-A差':>9}{'输入差L':>9}{'输入差A':>9}")
        print("  " + "-" * 88)

        for p, kind in items:
            img = PIL.Image.open(p).convert("RGB")
            x_ref = det.preprocess(img)

            s_ref = float(np.max(det._patch_scores(
                det._embed_tensor(x_ref)))) if False else None

            def score_of(x):
                e = det.sess.run(None, {det.in_name: x})[0]
                if e.ndim == 3:
                    e = e[0]
                ps = det._patch_scores(np.ascontiguousarray(e))
                return float(np.max(ps))

            s_ref = score_of(x_ref)

            res = {}
            for name, flag in (("L", cv2.INTER_LINEAR), ("A", cv2.INTER_AREA)):
                x = prep_cv2(img, roi, det.resize, det.isize,
                             det.mean, det.std, flag)
                d_in = float(np.abs(x - x_ref).max())
                res[name] = (score_of(x), d_in)

            print(f"  {os.path.basename(p)[:26]:<26}{kind:>5}{s_ref:9.3f}"
                  f"{res['L'][0] - s_ref:+9.3f}{res['A'][0] - s_ref:+9.3f}"
                  f"{res['L'][1]:9.3f}{res['A'][1]:9.3f}")

        print()

    print("=" * 92)
    print("  说明：cv2-L/A 差 = 用 cv2 预处理后的分数 − PIL 参考分数（阈值 cam1=1.05 / cam2=0.85）")
    print("=" * 92)


if __name__ == "__main__":
    main()
