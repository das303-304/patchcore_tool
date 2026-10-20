# -*- coding: utf-8 -*-
"""验证：把记忆库烘焙进 ONNX 图，让 onnxruntime 直接算 KNN 距离。

背景：C# 没有 numpy / FAISS，手写 784x8749x1024 的矩阵乘会慢一个数量级。
如果记忆库能作为 initializer 编进 ONNX，C# 就只需要 in/out 两次调用，
矩阵乘交给 onnxruntime 的 MLAS（多线程 sgemm）。

本脚本只测「检索段」的速度与数值一致性，不产出交付模型。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(HERE, "models_onnx", "皮料相机1")

REPEAT = 15


def build_scorer(bank, path):
    """d2 = |e|^2 - 2*e@B^T + |B|^2 ，取每行最小值。"""
    m, d = bank.shape
    n = 784                                   # patch 数（28x28）

    bt = numpy_helper.from_array(np.ascontiguousarray(bank.T), "bankT")   # (D, M)
    bnorm = numpy_helper.from_array((bank * bank).sum(1).astype(np.float32), "bnorm")

    nodes = [
        helper.make_node("Cast", ["embedding"], ["emb_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["emb_f", "emb_f"], ["e_sq"]),
        # opset 13+ 的 ReduceSum 把 axes 移到了输入（ReduceMin 在 17 仍是属性）
        helper.make_node("ReduceSum", ["e_sq", "axes1"], ["en"], keepdims=1),
        helper.make_node("MatMul", ["emb_f", "bankT"], ["dot"]),          # (N, M)
        helper.make_node("Mul", ["dot", "two"], ["dot2"]),
        helper.make_node("Sub", ["en", "dot2"], ["t1"]),                  # (N, M)
        helper.make_node("Add", ["t1", "bnorm"], ["d2"]),                 # (N, M)
        helper.make_node("ReduceMin", ["d2"], ["best"], axes=[1], keepdims=1),
        helper.make_node("Max", ["best", "zero"], ["patch_scores"]),
    ]
    inits = [
        bt, bnorm,
        numpy_helper.from_array(np.array([1], np.int64), "axes1"),
        numpy_helper.from_array(np.array(2.0, np.float32), "two"),
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
    ]
    graph = helper.make_graph(
        nodes, "patchcore_scorer",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [n, d])],
        [helper.make_tensor_value_info("patch_scores", TensorProto.FLOAT, [n, 1])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def main():
    bank = np.load(os.path.join(ONNX_DIR, "memory_bank.npy")).astype(np.float32)
    print(f"记忆库: {bank.shape[0]} x {bank.shape[1]}  "
          f"({bank.nbytes / 1024 / 1024:.1f} MB)")

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((784, bank.shape[1])).astype(np.float32)

    # ---- 参照：numpy 实现（与 onnx_infer.py 分块逻辑一致）----
    bt = np.ascontiguousarray(bank.T)
    bn = (bank * bank).sum(1).astype(np.float32)
    en = (emb * emb).sum(1, keepdims=True)

    def np_knn():
        best = np.full((emb.shape[0],), np.inf, dtype=np.float32)
        for i in range(0, bank.shape[0], 4096):
            d2 = en - 2.0 * (emb @ bt[:, i:i + 4096]) + bn[i:i + 4096][None, :]
            best = np.minimum(best, d2.min(axis=1))
        return np.maximum(best, 0.0)[:, None]

    for _ in range(3):
        np_knn()
    t = []
    for _ in range(REPEAT):
        s = time.perf_counter()
        ref = np_knn()
        t.append((time.perf_counter() - s) * 1000)
    np_ms = float(np.median(t))

    # ---- ONNX 版 ----
    p = os.path.join(HERE, "testimgs", "_scorer_probe.onnx")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    build_scorer(bank, p)
    print(f"临时模型大小: {os.path.getsize(p) / 1024 / 1024:.1f} MB")

    rows = []
    for name, opts in [
        ("ORT 默认线程", None),
        ("ORT 8 线程", 8),
    ]:
        so = ort.SessionOptions()
        if opts:
            so.intra_op_num_threads = opts
        sess = ort.InferenceSession(p, sess_options=so,
                                    providers=["CPUExecutionProvider"])
        sess.run(None, {"embedding": emb})
        t = []
        for _ in range(REPEAT):
            s = time.perf_counter()
            out = sess.run(None, {"embedding": emb})[0]
            t.append((time.perf_counter() - s) * 1000)
        rows.append((name, float(np.median(t)), float(np.abs(out - ref).max())))

    print()
    print("=" * 70)
    print(f"  检索段（784 patch × {bank.shape[0]} 向量 × {bank.shape[1]} 维）")
    print("=" * 70)
    print(f"  {'实现':<20}{'中位耗时':>12}{'与 numpy 最大差':>18}")
    print("  " + "-" * 66)
    print(f"  {'numpy 分块(现用)':<20}{np_ms:11.1f}ms{'—':>18}")
    for name, ms, err in rows:
        print(f"  {name:<20}{ms:11.1f}ms{err:18.2e}")
    print("=" * 70)

    os.remove(p)
    print("\n结论：ONNX 版若 ≈ numpy 水平，C# 侧就无需手写 SIMD 矩阵乘。")


if __name__ == "__main__":
    main()
