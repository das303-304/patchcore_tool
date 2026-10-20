# -*- coding: utf-8 -*-
"""重建 v2 模型：ROI 由 C# 裁剪后送入，图内只做「缩放到 224 → 归一化 → 骨干 → KNN」。

为什么改（相比 patchcore_e2e.onnx）：
  旧版把 ROI 作为常量 Slice 固化在图里，ROI 一变就要重导模型。
  用户要求 ROI 在新建工程时由自己画 → ROI 必须由 C# 裁剪后作为输入送进来。
  C# 用 Halcon 的整数裁剪（CropPart），无损、无重采样，不会引入跨语言偏差。

关键约束（ORT 1.17，工控机在用的版本）：
  **Resize 的目标尺寸必须是常量 initializer**，不能由算子动态计算。
  v2 直接缩放成 224×224（拉伸），尺寸常量，彻底规避该问题。
  代价：ROI 长宽比不等于 1 时会被拉伸；但记忆库用同一规则建，自洽。

流程：
  1. 导出 front.onnx：ROI 图 → embedding (784,1024)
  2. 用 front.onnx 提取全部 OK 图的特征（保证与推理完全同一条链路）
  3. coreset 采样得记忆库
  4. 导出 patchcore_roi.onnx：front + KNN(内嵌) + 热图
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "patchcore-inspection", "src"))

import numpy as np
import onnx
import onnxruntime as ort
import PIL.Image
import torch
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
OPSET = 18
ISIZE = 224
PATCH_GRID = 28
CORESET_PCT = 0.03

CASES = [("皮料相机1", "cam1"), ("皮料相机2", "cam2")]


def _f(name, value):
    return numpy_helper.from_array(np.array([value], dtype=np.float32), name)


def _i(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.int64), name)


def gauss_kernel(sigma=4.0, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return w.astype(np.float32), radius


def _load_base(site_dir):
    """载入旧版 model.onnx 的骨干，把它 input 改名。"""
    meta_path = os.path.join(site_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    base = onnx.load(os.path.join(site_dir, "model.onnx"))
    for node in base.graph.node:
        for k, nm in enumerate(node.input):
            if nm == "input":
                node.input[k] = "bb_in"
    assert any("bb_in" in n.input for n in base.graph.node), "骨干输入重命名失败"
    return base, meta


def build_front(site_dir, out_path):
    """ROI 图 uint8 (1,1,h,w) → embedding (784,1024)。"""
    base, meta = _load_base(site_dir)
    mean = np.array(meta["imagenet_mean"], np.float32).reshape(1, 3, 1, 1)
    std = np.array(meta["imagenet_std"], np.float32).reshape(1, 3, 1, 1)

    pre = [
        helper.make_node("Cast", ["gray"], ["g_f"], to=TensorProto.FLOAT),
        helper.make_node("Concat", ["g_f", "g_f", "g_f"], ["rgb"], axis=1),
        # 先拉伸到 256×256，再中心裁 224。
        # 为什么不直接缩到 224：实测直接缩会掉一档分辨率，弱缺陷（cam1 的 16_）
        # 从 0.618 掉到 0.528、检出从 10/10 变 9/10。256→裁 224 与旧版几何一致。
        # 两处尺寸都是常量，不触发 ORT 1.17 的动态 sizes 限制。
        helper.make_node(
            "Resize", ["rgb", "", "", "r_sizes"], ["rs"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            nearest_mode="round_prefer_floor",
            antialias=1,
        ),
        helper.make_node("Slice", ["rs", "c_st", "c_en", "c_ax"], ["crop"]),
        helper.make_node("Div", ["crop", "c255"], ["x01"]),
        helper.make_node("Sub", ["x01", "mean"], ["x_c"]),
        helper.make_node("Div", ["x_c", "std"], ["bb_in"]),
    ]

    nodes = pre + list(base.graph.node)
    for k, nd in enumerate(nodes):
        if not nd.name:
            nd.name = f"{nd.op_type}_{k}"

    inits = list(base.graph.initializer) + [
        _f("c255", 255.0),
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(std, "std"),
        _i("r_sizes", [1, 3, 256, 256]),
        _i("c_st", [16, 16]), _i("c_en", [16 + ISIZE, 16 + ISIZE]),
        _i("c_ax", [2, 3]),
    ]

    graph = helper.make_graph(
        nodes, "patchcore_front",
        [helper.make_tensor_value_info("gray", TensorProto.UINT8, [1, 1, "H", "W"])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [784, 1024])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


def build_final(site_dir, front_path, bank, out_path):
    """front + KNN(内嵌 bank) + 热图。"""
    front = onnx.load(front_path)
    bank = bank.astype(np.float32)
    gw, radius = gauss_kernel(4.0)

    post = [
        helper.make_node("Mul", ["embedding", "embedding"], ["e_sq"]),
        helper.make_node("ReduceSum", ["e_sq", "ax1"], ["en"], keepdims=1),
        helper.make_node("MatMul", ["embedding", "bankT"], ["dot"]),
        helper.make_node("Mul", ["dot", "two"], ["dot2"]),
        helper.make_node("Sub", ["en", "dot2"], ["t1"]),
        helper.make_node("Add", ["t1", "bnorm"], ["d2"]),
        helper.make_node("ReduceMin", ["d2", "ax1"], ["best"], keepdims=1),
        helper.make_node("Max", ["best", "zero"], ["patch_scores"]),
        helper.make_node("ReduceMax", ["patch_scores", "ax01"], ["score"], keepdims=1),
        helper.make_node("Reshape", ["patch_scores", "map_shape"], ["ps4"]),
        helper.make_node("Resize", ["ps4", "", "", "map_sizes"], ["up"],
                         mode="linear", coordinate_transformation_mode="align_corners"),
        helper.make_node("Pad", ["up", "pad_w"], ["ph"], mode="reflect"),
        helper.make_node("Conv", ["ph", "kw"], ["h1"]),
        helper.make_node("Pad", ["h1", "pad_h"], ["pv"], mode="reflect"),
        helper.make_node("Conv", ["pv", "kh"], ["anomaly_map"]),
    ]

    nodes = list(front.graph.node) + post
    for k, nd in enumerate(nodes):
        if not nd.name:
            nd.name = f"{nd.op_type}_{k}"

    inits = list(front.graph.initializer) + [
        numpy_helper.from_array(np.ascontiguousarray(bank.T), "bankT"),
        numpy_helper.from_array((bank * bank).sum(1).reshape(1, -1).astype(np.float32), "bnorm"),
        _f("two", 2.0), _f("zero", 0.0),
        _i("ax1", [1]), _i("ax01", [0, 1]),
        _i("map_shape", [1, 1, PATCH_GRID, PATCH_GRID]),
        _i("map_sizes", [1, 1, ISIZE, ISIZE]),
        numpy_helper.from_array(
            np.array([0, 0, 0, radius, 0, 0, 0, radius], np.int64), "pad_w"),
        numpy_helper.from_array(
            np.array([0, 0, radius, 0, 0, 0, radius, 0], np.int64), "pad_h"),
        numpy_helper.from_array(gw.reshape(1, 1, 1, -1), "kw"),
        numpy_helper.from_array(gw.reshape(1, 1, -1, 1), "kh"),
    ]

    graph = helper.make_graph(
        nodes, "patchcore_roi",
        [helper.make_tensor_value_info("gray", TensorProto.UINT8, [1, 1, "H", "W"])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1]),
         helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT,
                                       [1, 1, ISIZE, ISIZE])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


def roi_crop(path, roi):
    return np.asarray(PIL.Image.open(path).convert("L"), dtype=np.uint8)[
        roi[1]:roi[3], roi[0]:roi[2]]


def main():
    from patchcore.sampler import ApproximateGreedyCoresetSampler

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {dev}")

    for site, tag in CASES:
        site_dir = os.path.join(HERE, "models_onnx", site)
        with open(os.path.join(site_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        roi = meta["roi"]

        print(f"\n[{site}] 1/4 导出 front.onnx ...", flush=True)
        front_path = os.path.join(site_dir, "front.onnx")
        build_front(site_dir, front_path)

        print(f"[{site}] 2/4 提取全部 OK 图特征 ...", flush=True)
        cache = os.path.join(site_dir, f"_emb_cache_{tag}.npy")
        if os.path.exists(cache):
            all_emb = np.load(cache)
            print(f"    复用缓存 {all_emb.shape}", flush=True)
        else:
            sess = ort.InferenceSession(front_path, providers=["CPUExecutionProvider"])
            d = os.path.join(OK_ROOT, site)
            files = sorted(os.listdir(d))
            embs = []
            for i, f in enumerate(files):
                x = roi_crop(os.path.join(d, f), roi)[None, None]
                embs.append(sess.run(None, {"gray": x})[0])
                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{len(files)}", flush=True)
            all_emb = np.concatenate(embs, axis=0)
            np.save(cache, all_emb)
        print(f"    特征总数 {all_emb.shape[0]} × {all_emb.shape[1]}"
              f"  ({all_emb.nbytes / 1024 / 1024:.0f} MB)", flush=True)

        print(f"[{site}] 3/4 coreset 采样 pct={CORESET_PCT} ...", flush=True)
        sampler = ApproximateGreedyCoresetSampler(
            CORESET_PCT, dev, dimension_to_project_features_to=128)
        bank = sampler.run(
            torch.from_numpy(all_emb).to(dev))
        if isinstance(bank, torch.Tensor):
            bank = bank.cpu().numpy()
        bank = np.asarray(bank, dtype=np.float32)
        print(f"    记忆库 {bank.shape[0]} × {bank.shape[1]}"
              f"  ({bank.nbytes / 1024 / 1024:.1f} MB)", flush=True)

        print(f"[{site}] 4/4 导出 patchcore_roi.onnx ...", flush=True)
        out = os.path.join(site_dir, "patchcore_roi.onnx")
        build_final(site_dir, front_path, bank, out)
        np.save(os.path.join(site_dir, "memory_bank_v2.npy"), bank)
        print(f"    已保存 {out}  ({os.path.getsize(out) / 1024 / 1024:.1f} MB)")

    print("\n完成。")


if __name__ == "__main__":
    main()
