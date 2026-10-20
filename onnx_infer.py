"""PatchCore ONNX 推理 + 可视化（不依赖 PyTorch / FAISS）。

推理链：
    ONNX    图像张量 (1,3,224,224) -> patch 特征 (784,1024)
    numpy   特征 -> 到记忆库的最近邻平方 L2 距离 -> patch 分数
    numpy   图像分数 = patch 分数最大值
    numpy   28x28 -> 224x224 双线性 -> 高斯 σ=4 -> 异常热图
    cv2     热图叠加 + 最大异常区 bbox + 判定文字

关于「ONNX 只做特征提取」这件事：
    KNN 检索（原实现用 FAISS）走的是外部索引，不在计算图里，导不出来。
    所以 ONNX 负责骨干前向，检索用 numpy 实现 —— 这也是 C++ 部署时的标准拆法。
    记忆库存成 memory_bank.npy，部署端只需一个矩阵乘。

用法:
    # 单张
    python onnx_infer.py -m models_onnx/皮料相机1 -i xxx.jpg -o runs/out
    # 整个文件夹
    python onnx_infer.py -m models_onnx/皮料相机1 -i <目录> -o runs/out
    # 覆盖阈值
    python onnx_infer.py -m models_onnx/皮料相机1 -i <目录> -t 0.95
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import PIL.Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Result:
    path: str
    score: float
    label: int
    verdict: str
    peak: float
    ms: float


class OnnxPatchCore:
    def __init__(self, model_dir, threshold=None, providers=None, roi=None):
        self.dir = Path(model_dir)
        with open(self.dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)

        # ROI 以导出时的为准，允许外部覆盖（工具里调 ROI 调试用）
        self.roi = tuple(roi) if roi else tuple(self.meta["roi"])

        # ---- 一体化模型（建库工具产物）：单文件，输出 score + anomaly_map ----
        # 预处理（Resize 256 antialias → 中心裁 224 → 归一化 → 骨干 → KNN）全在图里，
        # Python 这边只需要把 ROI 灰度块原样喂进去 —— 和 C# 侧的用法完全一致。
        self.integrated = (self.dir / "patchcore_roi.onnx").exists()
        self.msize = tuple(self.meta.get("mask_size", (224, 224)))
        self.threshold = float(threshold if threshold is not None
                               else self.meta.get("threshold", 0.5))
        if self.integrated:
            self._init_integrated(providers)
            return

        self.resize = int(self.meta["resize"])
        self.isize = int(self.meta["input_size"])
        self.mean = np.asarray(self.meta["imagenet_mean"], np.float32)
        self.std = np.asarray(self.meta["imagenet_std"], np.float32)
        self.ph, self.pw = self.meta["patch_shape"]
        self.sigma = float(self.meta["gaussian_sigma"])
        self.msize = tuple(self.meta["mask_size"])
        self.n_neighbors = int(self.meta.get("n_neighbors", 1))

        self.threshold = float(threshold if threshold is not None
                               else self.meta["threshold"])

        bank = np.load(self.dir / "memory_bank.npy")
        bnorm = np.load(self.dir / "bank_norms.npy")
        self.bank = bank.astype(np.float32)
        self.bnorm = bnorm.astype(np.float32)

        # 只保留最近邻数=1 的快速路径；多于 1 时退化为 chunk 全排序
        self.bank_t = np.ascontiguousarray(self.bank.T)

        if providers is None:
            requests_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            providers = (["CUDAExecutionProvider"] if requests_cuda
                         else []) + ["CPUExecutionProvider"]
        else:
            requests_cuda = any("CUDA" in p for p in providers)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.dir / "model.onnx"),
                                         sess_options=so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        # 实际生效的 provider。请求 CUDA 但 CUDA/cuDNN 版本不匹配时，
        # onnxruntime 会**静默回退到 CPU** —— 必须如实记录，不能假装在用 GPU。
        self.providers_actual = self.sess.get_providers()
        if requests_cuda and "CUDAExecutionProvider" not in self.providers_actual:
            print("[警告] 请求 CUDA 但实际回退到 CPU。"
                  "常见原因：onnxruntime-gpu 与 CUDA/cuDNN 版本不匹配"
                  "（ORT 1.31 需要 CUDA 13，而 torch 2.6+cu124 只带 CUDA 12.4）")

    def _init_integrated(self, providers):
        if providers is None:
            requests_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            providers = (["CUDAExecutionProvider"] if requests_cuda
                         else []) + ["CPUExecutionProvider"]
        else:
            requests_cuda = any("CUDA" in p for p in providers)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(self.dir / "patchcore_roi.onnx"),
                                         sess_options=so, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.providers_actual = self.sess.get_providers()
        if requests_cuda and "CUDAExecutionProvider" not in self.providers_actual:
            print("[警告] 请求 CUDA 但实际回退到 CPU。")

    def _infer_integrated(self, img_path):
        """ROI 灰度块直接进图：输出 score 与 224×224 热图。"""
        img = PIL.Image.open(img_path)
        t0 = time.perf_counter()
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        x0, y0, x1, y1 = self.roi
        h, w = gray.shape[:2]
        x1, y1 = min(x1, w), min(y1, h)
        x0, y0 = max(0, min(x0, x1 - 1)), max(0, min(y0, y1 - 1))
        crop = np.ascontiguousarray(gray[y0:y1, x0:x1])[None, None]
        outs = self.sess.run(None, {self.in_name: crop})
        score = float(np.asarray(outs[0]).ravel()[0])
        amap = np.asarray(outs[1]).reshape(self.msize).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000
        label = int(score >= self.threshold)
        return score, label, amap, ms, img.convert("RGB")

    # ---------- 预处理：必须与训练完全一致 ----------
    def preprocess(self, img: PIL.Image):
        # 1) 裁 ROI（原图坐标系）
        im = img.crop(self.roi)
        # 2) 短边缩放到 resize(256)，保持长宽比
        #    严格对齐 torchvision T.Resize(int) 的算法：
        #      短边 = size；长边 = int(size * long / short)   ← 截断，不是四舍五入
        w, h = im.size
        if w <= h:
            nw, nh = self.resize, int(self.resize * h / w)
        else:
            nh, nw = self.resize, int(self.resize * w / h)
        im = im.resize((nw, nh), PIL.Image.BILINEAR)
        # 3) 中心裁剪 isize(224)，对齐 torchvision CenterCrop 的 round 取整
        left = int(round((nw - self.isize) / 2.0))
        top = int(round((nh - self.isize) / 2.0))
        im = im.crop((left, top, left + self.isize, top + self.isize))
        # 4) HWC uint8 -> NCHW float，ImageNet 归一化
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - self.mean) / self.std
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)

    # ---------- 距离检索 ----------
    def _patch_scores(self, emb: np.ndarray):
        """emb (N, D) -> 每个 patch 到记忆库的最近邻平方 L2 距离。"""
        e = emb.astype(np.float32)
        en = (e * e).sum(axis=1, keepdims=True)          # (N,1)
        nb = self.bank.shape[0]

        if self.n_neighbors == 1:
            # 分块检索。实测「一次性全量矩阵乘」中位速度相当（63~70ms），
            # 但尾延迟更差（~103ms）—— 因为每次要分配 (784, 8749) 约 27MB。
            # 生产看稳定性，选分块。
            best = np.full((e.shape[0],), np.inf, dtype=np.float32)
            step = 4096
            for i in range(0, nb, step):
                bn = self.bnorm[i:i + step][None, :]
                d2 = en - 2.0 * (e @ self.bank_t[:, i:i + step]) + bn
                best = np.minimum(best, d2.min(axis=1))
            return np.maximum(best, 0.0)

        # 多邻居：取前 k 个距离的均值（与原实现 np.mean(query_distances,-1) 一致）
        d2 = en - 2.0 * (e @ self.bank_t) + self.bnorm[None, :]
        k = self.n_neighbors
        part = np.partition(d2, k - 1, axis=1)[:, :k]
        return np.maximum(part.mean(axis=1), 0.0)

    # ---------- 后处理 ----------
    def postprocess(self, patch_scores: np.ndarray):
        ps = patch_scores.reshape(self.ph, self.pw)
        score = float(ps.max())                     # 图像分数 = patch 最大值
        m = cv2.resize(ps, self.msize, interpolation=cv2.INTER_LINEAR)
        # 与 scipy.ndimage.gaussian_filter(sigma=4) 等价：核半径取 4*sigma
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma,
                             borderType=cv2.BORDER_REFLECT)
        return score, m.astype(np.float32)

    # ---------- 主流程 ----------
    def infer(self, img_path):
        if self.integrated:
            return self._infer_integrated(img_path)
        img = PIL.Image.open(img_path).convert("RGB")
        t0 = time.perf_counter()
        x = self.preprocess(img)
        emb = self.sess.run(None, {self.in_name: x})[0]
        if emb.ndim == 3:
            emb = emb[0]
        ps = self._patch_scores(emb)
        score, amap = self.postprocess(ps)
        ms = (time.perf_counter() - t0) * 1000
        label = int(score >= self.threshold)
        return score, label, amap, ms, img

    def predict(self, img_path):
        """与 PyTorch 后端统一接口：返回 (score, mask)。"""
        score, _label, amap, _ms, _img = self.infer(img_path)
        return score, amap

    @property
    def backend_name(self):
        return "onnx"


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def colorize(amap, thr):
    """按阈值归一化配色：阈值以下整体压暗，只有超标区域才红。

    直接按 max 归一化会让高分图的正常区域也花花绿绿，看不出重点。
    这里用 score/thr 的相对值：0.55 倍阈值以下全暗，1.75 倍以上全红。
    """
    v = amap / max(thr, 1e-6)
    norm = np.clip((v - 0.55) / 1.2, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def render_panel(img, roi, amap, score, label, thr, title=""):
    """四栏：原图(带ROI框) / ROI料面 / 热图 / 叠加+bbox。"""
    x0, y0, x1, y1 = roi
    crop = np.asarray(img.crop(roi).convert("RGB"))
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    full = np.asarray(img.convert("RGB"))
    full_bgr = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
    hh, ww = full_bgr.shape[:2]
    sx, sy = ww / img.size[0], hh / img.size[1]
    cv2.rectangle(full_bgr, (int(x0 * sx), int(y0 * sy)),
                  (int(x1 * sx), int(y1 * sy)), (60, 60, 230), 3)

    heat = colorize(amap, thr)
    heat = cv2.resize(heat, (crop_bgr.shape[1], crop_bgr.shape[0]))

    # 热图叠加（只叠有响应的区域）
    ov = crop_bgr.copy()
    mask = heat.sum(axis=2) > 1
    ov[mask] = cv2.addWeighted(crop_bgr, 0.5, heat, 0.5, 0)[mask]

    # 最大异常区 bbox
    binm = (cv2.resize(amap, (crop_bgr.shape[1], crop_bgr.shape[0])) >
            max(thr * 0.6, amap.max() * 0.6)).astype(np.uint8)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (0, 0, 230) if label else (0, 180, 0)
    if cnts:
        x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if bw * bh > 0:
            cv2.rectangle(ov, (x, y), (x + bw, y + bh), color, 3)

    # 文字
    tag = f"{'NG' if label else 'OK'}  score={score:.3f}  thr={thr:.2f}"
    cv2.putText(ov, tag, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5,
                cv2.LINE_AA)
    cv2.putText(ov, tag, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                cv2.LINE_AA)

    # 拼成一行四栏，统一高度 480
    H = 480
    def fit(x):
        r = H / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * r), H))
    panels = [fit(full_bgr), fit(crop_bgr), fit(heat), fit(ov)]
    lab = ["original (ROI box)", "ROI crop", "anomaly map", "overlay"]
    for p, t in zip(panels, lab):
        cv2.putText(p, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (30, 30, 30), 1, cv2.LINE_AA)
    gap = np.full((H, 8, 3), 255, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, gap, p])
    return out


def main():
    ap = argparse.ArgumentParser(description="PatchCore ONNX 推理 + 可视化")
    ap.add_argument("-m", "--model", required=True, help="导出目录，如 models_onnx/皮料相机1")
    ap.add_argument("-i", "--input", required=True, help="单张图片或文件夹")
    ap.add_argument("-o", "--out", default="runs/onnx_infer")
    ap.add_argument("-t", "--threshold", type=float, default=None,
                    help="覆盖 meta.json 里的阈值")
    ap.add_argument("--no-render", action="store_true", help="不保存可视化图")
    args = ap.parse_args()

    det = OnnxPatchCore(args.model, args.threshold)
    inp = Path(args.input)
    files = ([inp] if inp.is_file()
             else sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT))
    if not files:
        raise SystemExit(f"没有找到图片: {inp}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"模型   : {args.model}")
    print(f"记忆库 : {det.bank.shape}  阈值 {det.threshold}")
    print(f"ROI    : {det.roi}   patch {det.ph}x{det.pw}  sigma {det.sigma}")
    print(f"provider: {det.sess.get_providers()}")
    print(f"待检   : {len(files)} 张\n")

    rows, ms_all = [], []
    for p in files:
        try:
            score, label, amap, ms, img = det.infer(str(p))
        except Exception as e:
            print(f"  [失败] {p.name}: {e}")
            continue
        ms_all.append(ms)
        verdict = "NG" if label else "OK"
        print(f"  {p.name:<34} score={score:7.3f}  {verdict}"
              f"  峰值={float(amap.max()):6.3f}  {ms:5.1f}ms")
        if not args.no_render:
            panel = render_panel(img, det.roi, amap, score, label, det.threshold)
            cv2.imwrite(str(out / f"{p.stem}_result.png"), panel)
        rows.append(Result(str(p), score, label, verdict, float(amap.max()), ms))

    n = len(rows)
    n_ng = sum(r.label for r in rows)
    print(f"\n汇总: {n} 张  NG {n_ng}  OK {n - n_ng}"
          f"  平均 {np.mean(ms_all):.1f} ms/张")

    csv_path = out / "results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "score", "verdict", "peak", "ms"])
        for r in rows:
            w.writerow([Path(r.path).name, f"{r.score:.4f}", r.verdict,
                        f"{r.peak:.4f}", f"{r.ms:.1f}"])
    print(f"CSV : {csv_path}")
    if not args.no_render:
        print(f"图  : {out}")


if __name__ == "__main__":
    main()
