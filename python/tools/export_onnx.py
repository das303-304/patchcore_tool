"""把已训练好的 PatchCore 记忆库导出为 ONNX + 记忆库文件。

为什么这么拆：
  PatchCore 的推理 = 骨干特征提取 + KNN 检索 + 后处理。
  KNN 检索（FAISS）不在计算图里，**无法导出到 ONNX**。
  所以导出方案是：
      ONNX 只负责：图像张量 -> patch 特征 (N, 1024)
      numpy 负责：特征 -> 最近邻距离 -> 异常分数 -> 热图
  这也是唯一能在不依赖 PyTorch/FAISS 的环境里跑通的做法。

产物（每个相机一个目录）：
    <out>/model.onnx          特征提取网络（batch=1, 输入 3x224x224）
    <out>/memory_bank.npy     记忆库特征 (N, 1024) float32
    <out>/bank_norms.npy      记忆库每行的 L2 范数平方（加速检索）
    <out>/meta.json           ROI / 阈值 / patch 形状 / 高斯 sigma 等

用法:
    python export_onnx.py                        # 导出 cam1 + cam2
    python export_onnx.py --site 皮料相机1
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

from infer_custom import LeatherInspector, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# 与 inspect_tool.py 保持一致
SITES = {
    "皮料相机1": dict(model="models/cam1", roi=(540, 30, 1780, 1270), threshold=1.05),
    "皮料相机2": dict(model="models/cam2", roi=(15, 30, 1400, 1270), threshold=0.85),
}


class FeatureExtractor(torch.nn.Module):
    """只做「图像 -> patch 特征」这一段，便于导出 ONNX。"""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, images):
        # detach=False 时 _embed 返回的是张量 (N_patch, D)，不是列表
        # （detach=True 才会因为列表推导变成 list of (D,)）
        feats = self.core._embed(images, detach=False)
        return feats                # (N_patch, D)


# ---------------------------------------------------------------------------
# ONNX 不支持 adaptive_avg_pool1d 的非整除输出尺寸
# （PatchCore 的 MeanMapper / Aggregator 都用了它）。
# 但导出时输入长度是常量，可以等价为一次固定矩阵乘法：
#   y[:, o] = mean(x[:, start:end])  ->  y = x @ W
# 这与 PyTorch 的 bin 划分完全一致，且 ONNX 原生支持 MatMul。
# ---------------------------------------------------------------------------
def _pool_matrix(L, O, dtype, device):
    import torch as _t
    W = np.zeros((L, O), np.float32)
    for i in range(O):
        start = (L * i) // O
        end = -((-(L * (i + 1))) // O)      # ceil
        end = max(end, start + 1)
        W[start:end, i] = 1.0 / (end - start)
    return _t.from_numpy(W).to(dtype=dtype, device=device)


class _OnnxFixups:
    """导出期的两处兼容处理（都不改变数学语义）：

    1. `F.adaptive_avg_pool1d` —— ONNX 不支持非整除输出尺寸。导出时输入长度是常量，
       等价替换为一次固定矩阵乘法。
    2. `Tensor.permute` 的负维度索引 —— 原代码用了 `permute(0, -3, -2, -1, 1, 2)`，
       ONNX 的 Transpose 只接受非负 perm，否则模型加载直接报 TypeInferenceError。
       这里统一转成正索引。
    """

    def __enter__(self):
        self._pool = torch.nn.functional.adaptive_avg_pool1d
        self._permute = torch.Tensor.permute

        def pool(input, output_size):
            if isinstance(output_size, (tuple, list)):
                output_size = output_size[0]
            L = input.size(-1)
            W = _pool_matrix(L, int(output_size), input.dtype, input.device)
            return (input.squeeze(1) @ W).unsqueeze(1)

        def permute(self, *dims):
            nd = self.dim()
            dims = tuple(d + nd if d < 0 else d for d in dims)
            return _orig_permute(self, *dims)

        _orig_permute = self._permute
        torch.nn.functional.adaptive_avg_pool1d = pool
        torch.Tensor.permute = permute
        return self

    def __exit__(self, *exc):
        torch.nn.functional.adaptive_avg_pool1d = self._pool
        torch.Tensor.permute = self._permute
        return False


def _check():
    """自检：MatMul 版池化与原生 adaptive_avg_pool1d 是否数值一致。"""
    torch.manual_seed(0)
    ok = True
    for L, O in [(4608, 1024), (9216, 1024), (3072, 768), (1000, 1024)]:
        x = torch.randn(4, 1, L)
        ref = torch.nn.functional.adaptive_avg_pool1d(x, O)
        W = _pool_matrix(L, O, x.dtype, x.device)
        got = (x.squeeze(1) @ W).unsqueeze(1)
        d = float((ref - got).abs().max())
        flag = d < 1e-5
        ok &= flag
        print(f"    池化自检 L={L:<5} O={O}  max|diff|={d:.2e}  "
              f"{'✓' if flag else '✗'}")
    return ok


def export_site(site, out_root, opset=14):
    cfg = SITES[site]
    model_dir = os.path.join(HERE, cfg["model"])
    roi = cfg["roi"]

    print(f"\n{'='*74}")
    print(f"[{site}]  模型 {cfg['model']}  ROI {roi}")
    print("=" * 74)

    det = LeatherInspector(model_dir, roi=roi)
    core = det.model
    # ONNX 导出放在 CPU 上最稳：CUDA 图里的某些算子 trace 会出问题
    core = core.to("cpu")
    core.eval()

    # 探测 patch 形状
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _, patch_shapes = core._embed(dummy, provide_patch_shapes=True)
    print(f"  patch 形状（多尺度）: {patch_shapes}")
    ph, pw = patch_shapes[0]

    # ---- 导出 ONNX ----
    out_dir = os.path.join(out_root, site)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")

    wrapper = FeatureExtractor(core).eval()
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"  ONNX 输出形状: {tuple(ref.shape)}")

    print("  池化等价性自检（MatMul vs adaptive_avg_pool1d）:")
    if not _check():
        raise SystemExit("池化替换后数值不一致，停止导出")

    with _OnnxFixups():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,          # 固定 batch=1，部署侧最省心
        )
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  已导出 {onnx_path}  ({size_mb:.1f} MB)")

    # 立即校验：能被 onnxruntime 加载，且输出形状正确
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
    print(f"  onnxruntime 加载校验: 输出 {out.shape}  {out.dtype}")
    if out.shape != tuple(ref.shape):
        raise SystemExit(f"输出形状不一致: {out.shape} vs {tuple(ref.shape)}")
    d = float(np.abs(out - ref.numpy()).max())
    print(f"  与 PyTorch 特征的最大绝对误差: {d:.3e}"
          f"  {'✓' if d < 1e-4 else '✗'}")

    # ---- 导出记忆库 ----
    idx = core.anomaly_scorer.nn_method.search_index
    n = idx.ntotal
    bank = idx.reconstruct_n(0, n).astype(np.float32)   # (N, 1024)
    np.save(os.path.join(out_dir, "memory_bank.npy"), bank)
    np.save(os.path.join(out_dir, "bank_norms.npy"),
            (bank * bank).sum(axis=1).astype(np.float32))

    # ---- meta ----
    meta = dict(
        site=site,
        model_src=cfg["model"],
        roi=list(roi),
        threshold=cfg["threshold"],
        input_size=224,
        resize=256,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        patch_shape=[ph, pw],
        embed_dim=int(bank.shape[1]),
        n_bank=int(n),
        n_neighbors=int(core.anomaly_scorer.n_nearest_neighbours),
        gaussian_sigma=float(core.anomaly_segmentor.smoothing),
        mask_size=list(core.anomaly_segmentor.target_size),
        score_reduce="max",          # 图像分数 = patch 分数最大值
        distance="squared_l2",       # FAISS IndexFlatL2 返回平方 L2，不开方
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  记忆库 {bank.shape} -> memory_bank.npy "
          f"({os.path.getsize(os.path.join(out_dir,'memory_bank.npy'))/1024/1024:.1f} MB)")
    print(f"  meta.json 已写")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="只导出某个站点")
    ap.add_argument("--out", default="models_onnx")
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    sites = [args.site] if args.site else list(SITES)
    out_root = os.path.join(HERE, args.out)
    for s in sites:
        export_site(s, out_root, args.opset)

    print(f"\n全部导出完成 -> {out_root}")
