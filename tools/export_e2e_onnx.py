# -*- coding: utf-8 -*-
"""导出一体化 ONNX：一张灰度图进去，分数 + 热图出来。

设计目标（为 C# 部署服务）：
    C# 侧只做一件事 —— 把 Halcon 图像的像素拷进 byte[]，然后 session.Run。
    图内完成：ROI 裁剪 → 缩放(带抗锯齿) → 中心裁剪 → 归一化 → 骨干
              → KNN(记忆库内嵌) → 分数 → 热图上采样 + 高斯平滑

为什么这么设计（实测依据，别改）：
  1. 缩放若放在 C# 侧用 OpenCvSharp INTER_LINEAR，正常料分数会从 0.48 跳到 1.31
     → 直接误报。ONNX Resize 的 antialias=1 与 PIL 等价（差 0.03）。
  2. KNN 若放在 C# 侧手写，784x8749x1024 约 7G 乘加，纯 C# 慢一个数量级。
     编进图里由 onnxruntime 的 MLAS 算，实测 37ms，比 numpy 还快。
  3. ROI 编进图里，C# 不用做坐标换算 —— 少一个出错环节。
  4. **缩放目标尺寸必须是常量**：ORT 1.17（工控机在用的版本）不支持由算子动态
     计算出来的 Resize sizes，会报 "Number of elements of input 'sizes' (0)"。
     ROI 尺寸本来就固定，所以直接算好写死。

输入输出：
    in : gray  uint8 (1,1,H,W)   —— ROI 之外的区域会被忽略，可传整幅图
    out: score float32 (1,1)     —— 图像级异常分（越大越差）
         anomaly_map float32 (1,1,224,224)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OPSET = 18          # antialias 需要 Resize-18
PATCH_GRID = 28
ISIZE = 224
RESIZE = 256


def _f(name, value):
    # 统一用 1 维 [1]，不要用 0 维标量：
    # Where(branch) 遇到标量会把输出推成 rank 0，导致后续 Concat/Reshape 的轴非法
    return numpy_helper.from_array(np.array([value], dtype=np.float32), name)


def _i(name, value):
    return numpy_helper.from_array(np.array(value, dtype=np.int64), name)


def gauss_kernel(sigma=4.0, truncate=4.0):
    """与 scipy.ndimage.gaussian_filter 的核一致：半径 int(truncate*sigma+0.5)。"""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return w.astype(np.float32), radius


def resize_and_crop_geometry(w, h):
    """复刻 torchvision Resize(int)+CenterCrop 的取整规则。"""
    if w <= h:
        nw, nh = RESIZE, int(RESIZE * h / w)      # 长边 int() 截断
    else:
        nh, nw = RESIZE, int(RESIZE * w / h)
    left = int(round((nw - ISIZE) / 2.0))          # round 取整
    top = int(round((nh - ISIZE) / 2.0))
    return nw, nh, left, top


def build(site_dir, out_path):
    with open(os.path.join(site_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    bank = np.load(os.path.join(site_dir, "memory_bank.npy")).astype(np.float32)
    base = onnx.load(os.path.join(site_dir, "model.onnx"))

    # 骨干图原本输入叫 input，改成内部名字，避免和新的整图输入混淆
    for node in base.graph.node:
        for k, nm in enumerate(node.input):
            if nm == "input":
                node.input[k] = "bb_in"
    assert any("bb_in" in n.input for n in base.graph.node), "骨干输入重命名失败"

    x0, y0, x1, y1 = meta["roi"]
    rw, rh = x1 - x0, y1 - y0
    nw, nh, left, top = resize_and_crop_geometry(rw, rh)

    mean = np.array(meta["imagenet_mean"], np.float32).reshape(1, 3, 1, 1)
    std = np.array(meta["imagenet_std"], np.float32).reshape(1, 3, 1, 1)
    gw, radius = gauss_kernel(meta["gaussian_sigma"])

    pre = [
        # 灰度 uint8 -> float -> 复制 3 通道（模型训练时是 R=G=B）
        helper.make_node("Cast", ["gray"], ["g_f"], to=TensorProto.FLOAT),
        helper.make_node("Concat", ["g_f", "g_f", "g_f"], ["rgb"], axis=1),
        # ROI 裁剪（常量坐标）
        helper.make_node("Slice", ["rgb", "roi_st", "roi_en", "roi_ax"], ["roi"]),
        # 缩放到常量尺寸，antialias=1 是重点（等价 PIL 抗锯齿下采样）
        helper.make_node(
            "Resize", ["roi", "", "", "r_sizes"], ["rs"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            nearest_mode="round_prefer_floor",
            antialias=1,
        ),
        # 中心裁剪 224（常量偏移）
        helper.make_node("Slice", ["rs", "cr_st", "cr_en", "cr_ax"], ["crop224"]),
        # 归一化
        helper.make_node("Div", ["crop224", "c255"], ["x01"]),
        helper.make_node("Sub", ["x01", "mean"], ["x_c"]),
        helper.make_node("Div", ["x_c", "std"], ["bb_in"]),
    ]

    post = [
        # 每个 patch 到记忆库的最近邻平方 L2
        helper.make_node("Mul", ["embedding", "embedding"], ["e_sq"]),
        helper.make_node("ReduceSum", ["e_sq", "ax1"], ["en"], keepdims=1),
        helper.make_node("MatMul", ["embedding", "bankT"], ["dot"]),
        helper.make_node("Mul", ["dot", "two"], ["dot2"]),
        helper.make_node("Sub", ["en", "dot2"], ["t1"]),
        helper.make_node("Add", ["t1", "bnorm"], ["d2"]),
        # opset18 起 ReduceMin/ReduceMax 的 axes 也移到了输入
        helper.make_node("ReduceMin", ["d2", "ax1"], ["best"], keepdims=1),
        helper.make_node("Max", ["best", "zero"], ["patch_scores"]),
        # 图像分数 = 所有 patch 的最大值（与原实现一致）
        helper.make_node("ReduceMax", ["patch_scores", "ax01"], ["score"], keepdims=1),
        # 热图：28x28 -> 224（align_corners，对应 F.interpolate）
        helper.make_node("Reshape", ["patch_scores", "map_shape"], ["ps4"]),
        helper.make_node("Resize", ["ps4", "", "", "map_sizes"], ["up"],
                         mode="linear", coordinate_transformation_mode="align_corners"),
        # 高斯平滑（可分离两次一维卷积）。仅用于显示，不影响 score。
        helper.make_node("Pad", ["up", "pad_w"], ["ph"], mode="reflect"),
        helper.make_node("Conv", ["ph", "kw"], ["h1"]),
        helper.make_node("Pad", ["h1", "pad_h"], ["pv"], mode="reflect"),
        helper.make_node("Conv", ["pv", "kh"], ["anomaly_map"]),
    ]

    inits = list(base.graph.initializer) + [
        numpy_helper.from_array(np.ascontiguousarray(bank.T), "bankT"),   # (D, M)
        numpy_helper.from_array((bank * bank).sum(1).reshape(1, -1).astype(np.float32), "bnorm"),
        _f("two", 2.0), _f("zero", 0.0), _f("c255", 255.0),
        numpy_helper.from_array(mean, "mean"), numpy_helper.from_array(std, "std"),
        # ROI 与裁剪的常量几何
        _i("roi_st", [y0, x0]), _i("roi_en", [y1, x1]), _i("roi_ax", [2, 3]),
        _i("r_sizes", [1, 3, nh, nw]),
        _i("cr_st", [top, left]), _i("cr_en", [top + ISIZE, left + ISIZE]),
        _i("cr_ax", [2, 3]),
        _i("ax1", [1]), _i("ax01", [0, 1]),
        _i("map_shape", [1, 1, PATCH_GRID, PATCH_GRID]),
        _i("map_sizes", [1, 1, ISIZE, ISIZE]),
        # ONNX Pad 的 pads 是 [N_begin,C_begin,H_begin,W_begin, N_end,C_end,H_end,W_end]
        numpy_helper.from_array(
            np.array([0, 0, 0, radius, 0, 0, 0, radius], np.int64), "pad_w"),
        numpy_helper.from_array(
            np.array([0, 0, radius, 0, 0, 0, radius, 0], np.int64), "pad_h"),
        numpy_helper.from_array(gw.reshape(1, 1, 1, -1), "kw"),
        numpy_helper.from_array(gw.reshape(1, 1, -1, 1), "kh"),
    ]

    nodes = pre + list(base.graph.node) + post
    # 给每个节点补名字，便于定位 ONNX 加载期报错（报错里的 Node 名）
    for k, nd in enumerate(nodes):
        if not nd.name:
            nd.name = f"{nd.op_type}_{k}"

    graph = helper.make_graph(
        nodes, "patchcore_e2e",
        [helper.make_tensor_value_info("gray", TensorProto.UINT8,
                                       [1, 1, "H", "W"])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1]),
         helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT,
                                       [1, 1, ISIZE, ISIZE])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return dict(roi_size=[rw, rh], resize_to=[nw, nh], crop_offset=[left, top])


def main():
    for site in ["皮料相机1", "皮料相机2"]:
        d = os.path.join(HERE, "models_onnx", site)
        out = os.path.join(d, "patchcore_e2e.onnx")
        print(f"[{site}] 构建中...", flush=True)
        info = build(d, out)
        print(f"  已保存 {out}  ({os.path.getsize(out) / 1024 / 1024:.1f} MB)")
        print(f"  ROI {info['roi_size']} -> 缩放 {info['resize_to']} "
              f"-> 裁剪偏移 {info['crop_offset']}")


if __name__ == "__main__":
    main()
