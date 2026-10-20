# -*- coding: utf-8 -*-
"""导出提速版 PatchCore 骨干模板 -> models_onnx/_base/model.onnx

与 _archive/export_onnx.py 的差别：只导出骨干，不需要记忆库语义、不写 meta。
真正被替换的只有「图像 -> patch 特征」这一段，记忆库与 KNN 由 build_tool 拼在外面。

为什么要重导（实测数据见 docs/plan-onnx-speedup-2026-10-19.md）：
  原 _embed 是「先拆 patch -> 再上采样 -> 最后求均值」，导出后变成
    逐 patch 上采样 43.5ms + 6 维 transpose 34.7ms + 两个稠密矩阵乘 46.7ms
  其中前三项共 118ms 是纯冗余。均值与双线性插值可交换，改成
  「先池化 -> 再上采样」，数值完全等价（实测 max|diff| = 8.94e-08）。

数值等价性的三条依据：
  1. layer3 的 3x3 均值池化与双线性上采样可交换（均为线性，且逐 patch 独立）
  2. layer3 MeanMapper 的 bin=9 恰是一个通道的 3x3 邻域 -> 就是 avg_pool2d(3,1,1)
  3. Aggregator 的 bin=2 等于「e2 前半两两平均 + e3 前半两两平均」后拼接
     （不是 (e2+e3)/2：reshape(784,2048) 后 e2 的 1024 维在前，前 512 个 bin 全落在 e2 内）

layer2 MeanMapper 必须保留原实现：bin=5 跨越通道边界、首尾重叠、末尾越界截断
（覆盖 5120 > 4608），是跨通道重叠滑动平均，没有等价的池化算子可表达。

用法:
    python _archive/export_fast_base.py                 # 导出并自检
    python _archive/export_fast_base.py --dry-run       # 只自检不写文件
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

# 本脚本在 _archive/ 下，项目根是上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "patchcore-inspection", "src"))

import torchvision.transforms as T  # noqa: E402
import patchcore.common  # noqa: E402
from patchcore import patchcore as pc  # noqa: E402

OUT_PATH = os.path.join(ROOT, "models_onnx", "_base", "model.onnx")
BANK_SRC = os.path.join(ROOT, "models", "cam1")
GRID = 28


# ---------------------------------------------------------------------------
# 核心：等价但更快的特征提取
# ---------------------------------------------------------------------------
class FastFeatureExtractor(torch.nn.Module):
    """与原 `_embed` 数值等价的快速实现，自适应不同骨干的通道数。

    三段优化，按通道数自动选择实现方式：

    **layer2**（不上采样，patch 数 784）
      通道 512 时 MeanMapper 的 bin 结构规整（每通道 2 个 5 邻域 bin，无一跨通道），
      可用「切片 + mean」把 3.7 GMAC 降到 4 MMAC；
      其它通道数退回固定矩阵乘（_OnnxFixups 生成），语义等价。

    **layer3**（需上采样 14x14 → 28x28）
      原始顺序是「patchify → 上采样到 784 张 → MeanMapper」。
      由于上采样作用在空间维、MeanMapper 作用在 (通道×邻域) 维，
      **两者可交换** ⇒ 改成「patchify → MeanMapper（只在 196 个 patch 上算）→ 上采样」，
      矩阵乘规模直接降 4 倍（行数 784 → 196），上采样对象恒为 1024 张图。
      通道 1024 时 bin=9 恰为好通道的 3x3 邻域，可进一步用 avg_pool2d 免掉矩阵乘。

    **Aggregator**：bin=2，等于各取前半两两平均后拼接（不是 (e2+e3)/2）。
    """

    def __init__(self, core, grid=GRID, layer2_mode="slice"):
        super().__init__()
        self.core = core
        self.grid = grid
        self.pm = core.patch_maker
        mods = core.forward_modules["preprocessing"].preprocessing_modules
        self.mm2, self.mm3 = mods[0], mods[1]
        self.layer2_mode = layer2_mode

    def forward(self, images):
        feats = self.core.forward_modules["feature_aggregator"](images)
        f2, f3 = feats["layer2"], feats["layer3"]

        # ---------- layer2 ----------
        p2, _ = self.pm.patchify(f2, return_spatial_info=True)   # (1,784,C2,3,3)
        nc2 = p2.shape[2]
        if self.layer2_mode == "slice":
            q = p2.reshape(-1, nc2, 9)
            a = q[:, :, 0:5].mean(dim=-1)
            b = q[:, :, 4:9].mean(dim=-1)
            e2 = torch.stack([a, b], dim=-1).reshape(-1, nc2 * 2)   # (784,1024)
        else:
            e2 = self.mm2(p2.reshape(-1, nc2 * 9))                  # (784,1024)

        # ---------- layer3 ----------
        p3, info3 = self.pm.patchify(f3, return_spatial_info=True)  # (1,196,C3,3,3)
        nc3 = p3.shape[2]
        h3, w3 = info3[0], info3[1]                                 # 14, 14
        if nc3 == 1024:
            # 通道 1024：MeanMapper 的 bin(9) 恰好是一个通道的 3x3 邻域 -> avg_pool2d
            buf = F.avg_pool2d(f3, 3, stride=1, padding=1)          # (1,1024,14,14)
        else:
            # 其它通道数：先在 196 个 patch 上做 MeanMapper，再上采样（顺序可交换）
            emb = self.mm3(p3.reshape(-1, nc3 * 9))                 # (196,1024)
            buf = emb.reshape(1, h3, w3, -1).permute(0, 3, 1, 2)    # (1,1024,14,14)
        u3 = F.interpolate(buf, size=(self.grid, self.grid),
                           mode="bilinear", align_corners=False)    # (1,1024,28,28)
        e3 = u3.permute(0, 2, 3, 1).reshape(-1, u3.shape[1])        # (784,1024)

        # ---------- Aggregator ----------
        e2h = e2.reshape(-1, e2.shape[1] // 2, 2).mean(-1)
        e3h = e3.reshape(-1, e3.shape[1] // 2, 2).mean(-1)
        return torch.cat([e2h, e3h], dim=1)


# ---------------------------------------------------------------------------
# ONNX 导出兼容（adaptive_avg_pool1d -> 固定矩阵乘）
# 与 _archive/export_onnx.py:64-110 等价，独立复制一份以免依赖老脚本的路径推导
# ---------------------------------------------------------------------------
def _pool_matrix(L, O, dtype, device):
    """y[:, o] = mean(x[:, start:end])  ->  y = x @ W，与 PyTorch bin 划分一致。"""
    W = np.zeros((L, O), np.float32)
    for i in range(O):
        start = (L * i) // O
        end = -((-(L * (i + 1))) // O)
        end = max(end, start + 1)
        W[start:end, i] = 1.0 / (end - start)
    return torch.from_numpy(W).to(dtype=dtype, device=device)


class _OnnxFixups:
    """导出期两处兼容处理（都不改变数学语义）：
    1. F.adaptive_avg_pool1d —— ONNX 不支持非整除输出尺寸，换成固定矩阵乘
    2. Tensor.permute 的负维度索引 —— ONNX Transpose 只接受非负 perm
    """

    def __enter__(self):
        self._pool = torch.nn.functional.adaptive_avg_pool1d
        self._permute = torch.Tensor.permute
        orig_permute = self._permute

        def pool(input, output_size):
            if isinstance(output_size, (tuple, list)):
                output_size = output_size[0]
            L = input.size(-1)
            W = _pool_matrix(L, int(output_size), input.dtype, input.device)
            return (input.squeeze(1) @ W).unsqueeze(1)

        def permute(self, *dims):
            nd = self.dim()
            dims = tuple(d + nd if d < 0 else d for d in dims)
            return orig_permute(self, *dims)

        torch.nn.functional.adaptive_avg_pool1d = pool
        torch.Tensor.permute = permute
        return self

    def __exit__(self, *exc):
        torch.nn.functional.adaptive_avg_pool1d = self._pool
        torch.Tensor.permute = self._permute
        return False


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def build_backbone(name):
    import torchvision.models as models
    table = {
        "wideresnet50": lambda: models.wide_resnet50_2(weights="IMAGENET1K_V1"),
        "resnet50": lambda: models.resnet50(weights="IMAGENET1K_V1"),
        "resnet34": lambda: models.resnet34(weights="IMAGENET1K_V1"),
        "resnet18": lambda: models.resnet18(weights="IMAGENET1K_V1"),
    }
    if name not in table:
        raise SystemExit("不支持的骨干: %s（可选 %s）" % (name, list(table)))
    return table[name]()


def load_core(backbone_name="wideresnet50", embed_dim=1024):
    """按名字构造 PatchCore（不需要记忆库，门禁 1 只比特征）。"""
    device = torch.device("cpu")
    core = pc.PatchCore(device)
    core.load(
        backbone=build_backbone(backbone_name).to(device).eval(),
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=[3, 224, 224],
        pretrain_embed_dimension=embed_dim,
        target_embed_dimension=embed_dim,
        patchsize=3, patchstride=1, anomaly_score_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    return core.eval()


def make_inputs(n=8):
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    files = sorted(glob.glob(os.path.join(ROOT, "ok_data", "*", "good", "*.jpg")))
    if not files:
        # 兜底：没有真实图就用随机张量（等价性验证仍然有效，只是弱一点）
        print("  [警告] 没找到 ok_data 样本，改用随机张量自检")
        return [torch.rand(1, 3, 224, 224) for _ in range(3)]
    xs = []
    for fp in files[:n]:
        xs.append(tf(PIL.Image.open(fp).convert("RGB")).unsqueeze(0))
    return xs


def gate1(core, wrapper, xs):
    """门禁 1：wrapper 与原始 _embed 数值等价。"""
    print("\n[门禁 1] PyTorch 层等价性（wrapper vs core._embed）")
    worst = 0.0
    for i, x in enumerate(xs):
        with torch.no_grad():
            ref = core._embed(x, detach=False)
            got = wrapper(x)
        d = float((ref - got).abs().max())
        worst = max(worst, d)
        print(f"  #{i}  max|diff|={d:.3e}  feature_max={float(ref.abs().max()):.3f}")
    ok = worst < 1e-5
    print(f"  >>> 最大绝对差 {worst:.3e}  {'✓ 通过' if ok else '✗ 失败'}")
    return ok


def gate2(onnx_path, xs, wrapper):
    """门禁 2：ONNX Runtime 输出与 PyTorch wrapper 逐元素一致。

    容差 1e-4 而不是 1e-7：ONNX 图里的 MeanMapper 被换成固定矩阵乘
    （_OnnxFixups._pool_matrix），求和顺序变了，与真 adaptive_avg_pool1d
    会有 1e-6 量级的浮点差异。这一级只看「导出有没有改变语义」。
    """
    import onnxruntime as ort
    print("\n[门禁 2] ONNX Runtime vs PyTorch wrapper")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    worst = 0.0
    for i, x in enumerate(xs):
        out = sess.run(None, {name: x.numpy()})[0]
        with torch.no_grad():
            ref = wrapper(x).numpy()
        d = float(np.abs(out - ref).max())
        worst = max(worst, d)
        if i == 0:
            print(f"  输出 shape={out.shape} dtype={out.dtype} "
                  f"幅值={float(np.abs(out).max()):.4f}")
        print(f"  #{i}  max|diff|={d:.3e}")
    ok = worst < 1e-4
    print(f"  >>> 最大绝对差 {worst:.3e}  {'✓ 通过' if ok else '✗ 失败'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="wideresnet50",
                    choices=["wideresnet50", "resnet50", "resnet34", "resnet18"])
    ap.add_argument("--dry-run", action="store_true", help="只自检，不写 onnx")
    ap.add_argument("--out", default="")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    print("=" * 72)
    print("  导出提速版 PatchCore 骨干   [%s]" % args.backbone)
    print("=" * 72)

    core = load_core(args.backbone)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        fs = core.forward_modules["feature_aggregator"](dummy)
    nc2, nc3 = fs["layer2"].shape[1], fs["layer3"].shape[1]
    mode = "slice" if nc2 == 512 else "matmul"
    print(f"  提取层: {core.layers_to_extract_from}")
    print(f"  layer2: {tuple(fs['layer2'].shape)}  layer3: {tuple(fs['layer3'].shape)}")
    print(f"  layer2 处理方式: {mode}   layer3 处理方式: "
          f"{'avg_pool2d(bin=9)' if nc3 == 1024 else 'MeanMapper@196patch + interp'}")

    wrapper = FastFeatureExtractor(core, layer2_mode=mode).eval()
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"  wrapper 输出: {tuple(ref.shape)}   (必须为 (784, 1024))")
    if tuple(ref.shape) != (784, 1024):
        raise SystemExit(f"输出形状不对: {tuple(ref.shape)}")

    xs = make_inputs(8)
    if not gate1(core, wrapper, xs):
        raise SystemExit("门禁 1 未通过，停止导出")

    if args.dry_run:
        print("\n--dry-run：不写文件")
        return

    out_path = args.out or (
        OUT_PATH if args.backbone == "wideresnet50"
        else os.path.join(ROOT, "models_onnx", "_bench", "%s.onnx" % args.backbone))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"\n  导出 -> {out_path}")
    t0 = time.time()
    with _OnnxFixups():
        torch.onnx.export(
            wrapper, dummy, out_path,
            input_names=["input"],          # build_tool.build_front 依赖此名
            output_names=["embedding"],     # 同上
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    print(f"  用时 {time.time()-t0:.1f}s  "
          f"大小 {os.path.getsize(out_path)/1048576:.1f} MB")

    if not gate2(out_path, xs, wrapper):
        raise SystemExit("门禁 2 未通过")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
