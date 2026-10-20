# -*- coding: utf-8 -*-
"""PatchCore 建库工具（本地网页版）。

把原来写死路径的 build_v2.py + calibrate_v2.py 变成一个可交互工具：
    选 OK 图目录 -> 图上画 ROI -> 建库 -> 输出 patchcore_roi.onnx + meta_v2.json + 建议阈值

与老脚本的差别：
  * 相机编号、OK 图目录、ROI 全部现场指定，不改源码
  * 记忆库规模按「向量数」指定，不再按 coreset 百分比猜
  * coreset 用等价但省内存的实现（官方版保留 N x k 距离矩阵，291k 点上要 10GB，
    本机 6GB 显卡必 OOM）
  * 建库完成后自动标定（全量 OK 图），可选用缺陷图验证检出率
  * 可选一键复制到 XVision 的 PatchCore/camN/

启动：
    .venv/Scripts/python.exe build_tool.py --open
或双击 启动建库工具.bat
"""
import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TQDM_DISABLE", "1")
# 缓解 CUDA 缓存碎片（coreset 循环几千轮，6GB 卡很容易碎到 OOM）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import PIL.Image
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
OPSET = 18
ISIZE = 224
PATCH_GRID = 28
RESIZE = 256           # 先拉伸 256 再中心裁 224，与线上模型几何一致
BASE_DIR = os.path.join(HERE, "models_onnx", "_base")

# 可选骨干。不同骨干的特征空间与**分数尺度完全不同**（resnet50 阈值约 0.9，
# resnet34 约 11，resnet18 约 6），所以：
#   1. 换骨干必须重建库 + 重标阈值，旧库/旧阈值一律作废
#   2. meta_v2.json 里记录 backbone，部署时与目标相机比对，不一致就告警
BACKBONES = {
    "resnet50":     dict(file="model.resnet50.onnx",     note="快，推荐"),
    "wideresnet50": dict(file="model.wideresnet50.onnx", note="慢，特征更强"),
    "resnet34":     dict(file="model.resnet34.onnx",     note="较快，判别力居中"),
    "resnet18":     dict(file="model.resnet18.onnx",     note="最快，判别力最弱"),
}
DEFAULT_BACKBONE = "resnet50"


def resolve_base(name=None):
    """骨干名 -> 模板路径。未知名字回退默认，避免建库中途才失败。"""
    key = name if name in BACKBONES else DEFAULT_BACKBONE
    return os.path.join(BASE_DIR, BACKBONES[key]["file"])


def cam_backbone(cam):
    """读某相机当前模型用的骨干；读不到返回空串。"""
    p = os.path.join(OUT_ROOT, "cam%d" % cam, "meta_v2.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("backbone", "") or ""
    except Exception:
        return ""


BASE_MODEL = resolve_base()      # 默认骨干路径（rebuild_cam.py 也用它）
OUT_ROOT = os.path.join(HERE, "models_onnx")
CFG_PATH = os.path.join(HERE, "build_tool_config.json")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 建库核心
# ---------------------------------------------------------------------------


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


def build_front(base_path, out_path):
    """ROI 灰度图 uint8 (1,1,h,w) -> embedding (784,1024)。

    尺寸常量写死（256 拉伸 + 中心裁 224），因为工控机的 ORT 1.17
    不支持由算子动态算出的 Resize sizes。
    """
    base = onnx.load(base_path)
    renamed = False
    for node in base.graph.node:
        for k, nm in enumerate(node.input):
            if nm == "input":
                node.input[k] = "bb_in"
                renamed = True
    if not renamed:
        raise RuntimeError(
            "骨干模板里找不到名为 input 的输入，不是标准 PatchCore 骨干: " + base_path)

    mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)

    pre = [
        helper.make_node("Cast", ["gray"], ["g_f"], to=TensorProto.FLOAT),
        helper.make_node("Concat", ["g_f", "g_f", "g_f"], ["rgb"], axis=1),
        helper.make_node(
            "Resize", ["rgb", "", "", "r_sizes"], ["rs"],
            mode="linear", coordinate_transformation_mode="half_pixel",
            nearest_mode="round_prefer_floor", antialias=1),
        helper.make_node("Slice", ["rs", "c_st", "c_en", "c_ax"], ["crop"]),
        helper.make_node("Div", ["crop", "c255"], ["x01"]),
        helper.make_node("Sub", ["x01", "mean"], ["x_c"]),
        helper.make_node("Div", ["x_c", "std"], ["bb_in"]),
    ]
    nodes = pre + list(base.graph.node)
    for k, nd in enumerate(nodes):
        if not nd.name:
            nd.name = "%s_%d" % (nd.op_type, k)

    inits = list(base.graph.initializer) + [
        _f("c255", 255.0),
        numpy_helper.from_array(mean, "mean"),
        numpy_helper.from_array(std, "std"),
        _i("r_sizes", [1, 3, RESIZE, RESIZE]),
        _i("c_st", [16, 16]),
        _i("c_en", [16 + ISIZE, 16 + ISIZE]),
        _i("c_ax", [2, 3]),
    ]
    graph = helper.make_graph(
        nodes, "patchcore_front",
        [helper.make_tensor_value_info("gray", TensorProto.UINT8, [1, 1, "H", "W"])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT,
                                       [PATCH_GRID * PATCH_GRID, 1024])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


def build_final(front_path, bank, out_path):
    """front + KNN(记忆库内嵌) + 热图。"""
    front = onnx.load(front_path)
    bank = np.ascontiguousarray(bank, dtype=np.float32)
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
            nd.name = "%s_%d" % (nd.op_type, k)

    inits = list(front.graph.initializer) + [
        numpy_helper.from_array(np.ascontiguousarray(bank.T), "bankT"),
        numpy_helper.from_array((bank * bank).sum(1).reshape(1, -1).astype(np.float32),
                                "bnorm"),
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
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


def _pairwise(a, b, chunk=32768):
    """分块欧氏距离，与官方 sampler 同公式。

    必须分块：全量 291k 点 x 128 维一次算要 129MB 临时张量，循环几千轮会把
    6GB 显卡的缓存分配器碎片化到 OOM（实测踩过）。
    """
    bb = (b * b).sum(1).unsqueeze(0)
    parts = []
    for i in range(0, a.shape[0], chunk):
        ac = a[i:i + chunk]
        aa = (ac * ac).sum(1, keepdim=True)
        parts.append((aa - 2.0 * ac.mm(b.T) + bb).clamp_(min=0).sqrt())
    return torch.cat(parts, dim=0)


MAX_CANDIDATES = 400000


def greedy_coreset(embs, target, on_progress=None):
    """近似贪心 coreset，等价于 ApproximateGreedyCoresetSampler 但省内存。

    两处与官方实现的差别（都是内存原因，不影响语义）：
      1. 官方每轮 cat 一列，最终保留 N x k 距离矩阵（291k x 8750 = 10GB）→ 必 OOM。
         这里维护「当前最小距离」向量逐轮 minimum，与「保留所有列再取 min」等价。
      2. 候选点超过 40 万时才抽样（当前 372 张 = 29 万，走全量）。
         实测抽到 10 万会让库覆盖面变差：自评最大分从 0.57 涨到 0.74。
    """
    n, dim = embs.shape
    if n > MAX_CANDIDATES:
        cand = np.sort(np.random.choice(n, MAX_CANDIDATES, replace=False))
        sub = np.ascontiguousarray(embs[cand])
    else:
        sub = embs
    m = sub.shape[0]
    k = int(max(1, min(target, m)))
    # 必须 no_grad：不开的话每轮距离计算都会挂 autograd 图，
    # 几千轮下来显存涨到 19GB 直接 OOM（实测踩过，官方 sampler 里也有这段）
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(sub, dtype=np.float32)).to(DEVICE)
        # 与官方一致：先去 128 维随机投影再算距离
        red = torch.nn.Linear(dim, 128, bias=False).to(DEVICE)(x)
        del x
        starts = np.random.choice(m, min(10, m), replace=False).tolist()
        anchor = _pairwise(red, red[starts]).mean(dim=1, keepdim=True)
        idx = np.empty(k, dtype=np.int64)
        for i in range(k):
            sel = int(torch.argmax(anchor).item())
            idx[i] = sel
            dd = _pairwise(red, red[sel:sel + 1])
            anchor = torch.minimum(anchor, dd)
            del dd
            if on_progress and (i % 100 == 0 or i == k - 1):
                on_progress((i + 1) / float(k))
        del red, anchor
    out = np.ascontiguousarray(sub[idx], dtype=np.float32)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def knn_scores(feats, bank, batch=4):
    """feats: (n, 784, D) -> (n,)，与模型里 ReduceMax(ReduceMin(d2)) 完全等价。

    用途是「不导出模型就算分」，留出标定要跑很多遍，走 ONNX 太慢。
    """
    b = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32)).to(DEVICE)
    bnorm = (b * b).sum(1).reshape(1, 1, -1)
    out = np.empty(len(feats), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            chunk = np.ascontiguousarray(feats[i:i + batch], dtype=np.float32)
            e = torch.from_numpy(chunk).to(DEVICE)
            en = (e * e).sum(-1, keepdim=True)
            d2 = (en - 2.0 * torch.matmul(e, b.T) + bnorm).clamp_(min=0)
            out[i:i + batch] = d2.min(-1).values.max(-1).values.cpu().numpy()
    del b
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def crop_roi(path, roi):
    g = np.asarray(PIL.Image.open(path).convert("L"), dtype=np.uint8)
    return g[roi[1]:roi[3], roi[0]:roi[2]]


def list_images(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMG_EXT:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# 后台任务
# ---------------------------------------------------------------------------
JOB_LOCK = threading.Lock()
JOB = dict(state="idle", stage="", pct=0, logs=[], cursor=0, result=None,
           error=None, t0=0.0, folder="", backbone="")


def log(text, cls=None):
    with JOB_LOCK:
        JOB["logs"].append(dict(text=text, cls=cls))
        if len(JOB["logs"]) > 4000:
            JOB["logs"] = JOB["logs"][-2000:]
            JOB["cursor"] = max(0, JOB["cursor"] - 2000)


def stage(text, pct):
    with JOB_LOCK:
        JOB["stage"] = text
        JOB["pct"] = float(pct)


def _save_cfg(cfg):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_cfg():
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def start_build(cam, folder, roi, target, ng_dir, deploy_dir, backbone=DEFAULT_BACKBONE):
    if backbone not in BACKBONES:
        raise RuntimeError("未知骨干: %s（可选 %s）" % (backbone, ", ".join(BACKBONES)))
    with JOB_LOCK:
        if JOB["state"] == "running":
            raise RuntimeError("已有建库任务在跑")
        JOB.update(state="running", stage="准备", pct=0, logs=[], cursor=0,
                   result=None, error=None, t0=time.time(), folder=folder)
        JOB["backbone"] = backbone

    def worker():
        try:
            _run_build(cam, folder, roi, int(target), ng_dir, deploy_dir, backbone)
        except Exception:
            msg = traceback.format_exc()
            log(msg, "err")
            try:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            with JOB_LOCK:
                JOB["state"] = "error"
                JOB["error"] = msg.strip().splitlines()[-1]
                JOB["pct"] = 0
                JOB["stage"] = "失败"
            _save_cfg(dict(cam=cam, folder=folder, roi=roi, target=target,
                           ng_dir=ng_dir, deploy_dir=deploy_dir,
                           last_error=JOB["error"]))

    threading.Thread(target=worker, daemon=True).start()


def resolve_deploy_dir(selected, cam):
    """把用户选的目录解析成真正的目标目录。

    用户可能停在三种层级，都要能用：
      1. <exe目录>/PatchCore/camN  → 本身就是目标
      2. <exe目录>/PatchCore       → 补 camN
      3. <exe目录> 或它的任意子目录 → 往上找 XVision.exe，再补 PatchCore/camN
    返回 (目标目录, 根目录, 错误信息)。
    """
    p = os.path.normpath((selected or "").replace("\\", "/"))
    if not p or not os.path.isdir(p):
        return None, None, "目录不存在：" + (selected or "")
    base = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    cam_name = "cam%d" % cam

    if base == cam_name.lower() and parent == "patchcore":
        return p, os.path.dirname(os.path.dirname(p)), ""
    if base == "patchcore":
        return os.path.join(p, cam_name), os.path.dirname(p), ""

    # 往上找（选到了 PatchCore/x、bin 之类）
    cur = p
    for _ in range(6):
        if os.path.exists(os.path.join(cur, "XVision.exe")):
            return os.path.join(cur, "PatchCore", cam_name), cur, ""
        up = os.path.dirname(cur)
        if up == cur:
            break
        cur = up

    # 往下找（用户很可能直接选项目根目录，如 E:/ATK/转盘机视觉）
    import glob
    for pat in ("XVision.exe", "*/XVision.exe",
                "*/bin/Debug/*/XVision.exe", "*/bin/Release/*/XVision.exe"):
        hits = sorted(glob.glob(os.path.join(p, pat)))
        if hits:
            root = os.path.dirname(hits[0])
            return os.path.join(root, "PatchCore", cam_name), root, ""

    return None, None, ("在 %s 及上下级都没找到 XVision.exe，"
                        "请选到 XVision.exe 所在目录（bin\\Debug\\net8.0-windows）" % p)


def _run_build(cam, folder, roi, target, ng_dir, deploy_dir, backbone=DEFAULT_BACKBONE):
    t0 = time.time()
    files = list_images(folder)
    if not files:
        raise RuntimeError("目录里没有图片: " + folder)
    x1, y1, x2, y2 = [int(v) for v in roi]
    if x2 - x1 < 32 or y2 - y1 < 32:
        raise RuntimeError("ROI 太小: %d x %d" % (x2 - x1, y2 - y1))

    base_model = resolve_base(backbone)
    if not os.path.exists(base_model):
        raise RuntimeError(
            "骨干模板不存在: %s\n  请先用 _archive/export_fast_base.py --backbone %s 生成"
            % (base_model, backbone))

    out_dir = os.path.join(OUT_ROOT, "cam%d" % cam)
    os.makedirs(out_dir, exist_ok=True)
    log("相机 cam%d，OK 图 %d 张" % (cam, len(files)), "dim")
    log("ROI = [%d, %d, %d, %d]  (%d x %d)"
        % (x1, y1, x2, y2, x2 - x1, y2 - y1), "dim")
    log("设备: %s" % DEVICE)

    prev = cam_backbone(cam)
    if prev and prev != backbone:
        log("⚠ 该相机现有模型用的是 %s，本次改用 %s" % (prev, backbone), "err")
        log("  分数尺度完全不同（阈值不可沿用），部署后必须重新标定阈值", "err")
    log("骨干: %s  (%s)" % (backbone, BACKBONES[backbone]["note"]), "dim")

    # ---- 1. 骨干 + front ----
    stage("导出特征提取图", 2)
    front_path = os.path.join(out_dir, "front.onnx")
    build_front(base_model, front_path)
    log("front.onnx 已生成 (%.0f MB)" % (os.path.getsize(front_path) / 1048576.0))

    # ---- 2. 提特征 ----
    sess = ort.InferenceSession(front_path, providers=["CPUExecutionProvider"])
    embs = []
    t_feat = time.time()
    for i, name in enumerate(files):
        x = crop_roi(os.path.join(folder, name), (x1, y1, x2, y2))[None, None]
        embs.append(sess.run(None, {"gray": x})[0])
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            stage("提取特征 %d/%d" % (i + 1, len(files)), 2 + 36.0 * (i + 1) / len(files))
    embs = np.concatenate(embs, axis=0)
    del sess
    log("特征 %d x %d (%.0f MB)  用时 %.1fs"
        % (embs.shape[0], embs.shape[1], embs.nbytes / 1048576.0,
           time.time() - t_feat))

    # ---- 3. 留出标定：只拿 80% 图建临时库，给未入库的 20% 打分 ----
    # 为什么必须留出：建库图的特征就在记忆库里，分数被人为压低。
    # 用它们自评会低估正常上界，阈值偏乐观 → 现场漏检。
    n = len(files)
    ppp = PATCH_GRID * PATCH_GRID

    def _take(idx):
        rows = (np.asarray(idx)[:, None] * ppp + np.arange(ppp)[None, :]).ravel()
        return embs[rows]

    hold = 0
    if n >= 20:
        hold = int(min(max(10, round(n * 0.2)), 50, n - 10))
    hold_mean = hold_std = hold_max = hold_thr = 0.0
    hold_idx = []
    if hold >= 10:
        stage("留出标定（80% 建临时库）", 40)
        perm = np.random.permutation(n)
        hold_idx = np.sort(perm[:hold])
        tr_idx = np.sort(perm[hold:])
        log("留出标定: 建库 %d 张 / 未入库 %d 张" % (len(tr_idx), hold))
        tgt_h = max(300, int(target * len(tr_idx) / float(n)))

        def _prog_h(p):
            stage("留出库 coreset %.0f%%" % (p * 100), 40 + 16.0 * p)

        bank_h = greedy_coreset(_take(tr_idx), tgt_h, _prog_h)
        hs = knn_scores(_take(hold_idx).reshape(hold, ppp, -1), bank_h)
        del bank_h
        hold_mean, hold_std = float(hs.mean()), float(hs.std())
        hold_max = float(hs.max())
        hold_thr = round(max(hold_mean + 6 * hold_std, hold_max * 1.02), 2)
        log("未入库正常图: 均值 %.4f  std %.4f  最大 %.4f（%s）"
            % (hold_mean, hold_std, hold_max,
               files[hold_idx[int(np.argmax(hs))]]))
    else:
        log("样本少于 20 张，跳过留出标定 —— 阈值只能按建库图自评，会偏乐观", "err")

    # ---- 4. coreset（全量） ----
    stage("coreset 采样（全量）", 58)
    log("coreset 目标 %d 向量（候选 %d）" % (target, len(embs)))
    t_cs = time.time()

    def cs_progress(p):
        stage("coreset 采样 %.0f%%" % (p * 100), 58 + 14.0 * p)

    bank = greedy_coreset(embs, target, cs_progress)
    log("记忆库 %d x %d (%.1f MB)  用时 %.1fs"
        % (bank.shape[0], bank.shape[1], bank.nbytes / 1048576.0, time.time() - t_cs))

    # ---- 5. 导出模型 ----
    stage("导出 patchcore_roi.onnx", 73)
    model_path = os.path.join(out_dir, "patchcore_roi.onnx")
    build_final(front_path, bank, model_path)
    np.save(os.path.join(out_dir, "memory_bank_v2.npy"), bank)
    model_mb = os.path.getsize(model_path) / 1048576.0
    log("模型已导出 %.0f MB  %s" % (model_mb, model_path))

    # ---- 6. 建库图自评（仅作参考，不用于定阈值） ----
    stage("建库图自评", 78)
    t_cal = time.time()
    self_scores = knn_scores(embs.reshape(n, ppp, -1), bank)
    del embs
    ms = (time.time() - t_cal) / n * 1000.0
    s_mean, s_std = float(self_scores.mean()), float(self_scores.std())
    s_max = float(self_scores.max())
    log("建库图自评: 均值 %.4f  std %.4f  最大 %.4f（特征在库内，仅作参考）"
        % (s_mean, s_std, s_max), "dim")

    # 阈值优先用留出结果；没有留出时退回自评并已警告
    if hold >= 10:
        thr = hold_thr
        log("阈值取自未入库图: max(均值+6σ, 最大×1.02) = %.2f" % thr, "ok")
    else:
        thr = round(max(s_mean + 6 * s_std, s_max * 1.02), 2)
        log("阈值退回建库图自评 = %.2f（偏乐观，建议补足 50 张以上样本）" % thr, "err")

    # ---- 7. ONNX 冒烟：确认导出的模型与 numpy 算分一致 ----
    stage("ONNX 一致性检查", 84)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    probe = list(range(min(5, n)))
    diffs = []
    for i in probe:
        x = crop_roi(os.path.join(folder, files[i]), (x1, y1, x2, y2))[None, None]
        onnx_s = float(sess.run(None, {"gray": x})[0].ravel()[0])
        diffs.append(abs(onnx_s - float(self_scores[i])))
    log("ONNX 与 numpy 算分最大差 %.2e（%d 张抽检）" % (max(diffs), len(diffs)),
        "ok" if max(diffs) < 1e-3 else "err")

    # ---- 8. 可选：NG 验证（走导出模型，端到端） ----
    ng_hit = ng_total = 0
    ng_scores = {}
    if ng_dir and os.path.isdir(ng_dir):
        ng_files = list_images(ng_dir)
        for i, name in enumerate(ng_files):
            x = crop_roi(os.path.join(ng_dir, name), (x1, y1, x2, y2))[None, None]
            ng_scores[name] = round(float(sess.run(None, {"gray": x})[0].ravel()[0]), 4)
            if (i + 1) % 10 == 0:
                stage("NG 验证 %d/%d" % (i + 1, len(ng_files)), 92)
        ng_total = len(ng_files)
        ng_hit = sum(1 for v in ng_scores.values() if v >= thr)
        log("NG 检出 %d/%d" % (ng_hit, ng_total),
            "ok" if ng_hit == ng_total else "err")
        miss = {k: v for k, v in ng_scores.items() if v < thr}
        if miss:
            log("未检出: %s" % json.dumps(miss, ensure_ascii=False), "dim")
    del sess

    # ---- 9. meta ----
    meta_v2 = dict(site="cam%d" % cam, model="patchcore_roi.onnx",
                   backbone=backbone,
                   roi=[x1, y1, x2, y2], n_ok=n,
                   threshold=thr, n_bank=int(bank.shape[0]),
                   holdout=dict(n=hold, mean=round(hold_mean, 6),
                                std=round(hold_std, 6), max=round(hold_max, 6)),
                   self_eval=dict(mean=round(s_mean, 6), std=round(s_std, 6),
                                  max=round(s_max, 6)),
                   ng_hit=ng_hit, ng_total=ng_total, ng_scores=ng_scores,
                   ms_per_image=round(ms, 1),
                   input_size=ISIZE, resize=RESIZE)
    with open(os.path.join(out_dir, "meta_v2.json"), "w", encoding="utf-8") as f:
        json.dump(meta_v2, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(site="cam%d" % cam, roi=[x1, y1, x2, y2],
                       input_size=ISIZE, resize=RESIZE,
                       imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD,
                       patch_shape=[PATCH_GRID, PATCH_GRID], embed_dim=1024,
                       n_bank=int(bank.shape[0]), n_neighbors=1,
                       gaussian_sigma=4.0, mask_size=[ISIZE, ISIZE],
                       score_reduce="max", distance="squared_l2"),
                  f, ensure_ascii=False, indent=2)

    # ---- 8. 可选部署 ----
    deployed = ""
    deploy_err = ""
    if deploy_dir:
        dst_dir, _root, err = resolve_deploy_dir(deploy_dir, cam)
        if err:
            deploy_err = err
            log("✗ " + deploy_err, "err")
            log("  模型已建好，只是没复制。也可以手动拷：", "dim")
            log("  %s  →  <XVision目录>\\PatchCore\\cam%d\\" % (model_path, cam), "dim")
        else:
            try:
                os.makedirs(dst_dir, exist_ok=True)
                old_meta = os.path.join(dst_dir, "meta_v2.json")
                if os.path.exists(old_meta):
                    try:
                        with open(old_meta, encoding="utf-8") as fh:
                            old_bb = json.load(fh).get("backbone", "")
                    except Exception:
                        old_bb = ""
                    if old_bb and old_bb != backbone:
                        log("⚠ 目标相机原用 %s，本次为 %s —— 模型与阈值将一并覆盖"
                            % (old_bb, backbone), "err")
                        log("  两份分数尺度不同，请确认这是有意切换骨干", "err")
                dst = os.path.join(dst_dir, "patchcore_roi.onnx")
                stage("复制到 XVision", 98)
                shutil.copy2(model_path, dst)
                shutil.copy2(os.path.join(out_dir, "meta_v2.json"),
                             os.path.join(dst_dir, "meta_v2.json"))
                deployed = dst
                log("已复制到 %s" % dst, "ok")
            except Exception as ex:
                deploy_err = "复制失败（%s），XVision 可能正开着占用文件" % ex
                log("✗ " + deploy_err, "err")

    stage("完成", 100)
    result = dict(model_path=model_path, out_dir=out_dir, model_mb=round(model_mb, 1),
                  backbone=backbone,
                  n_ok=n, n_bank=int(bank.shape[0]), threshold=thr,
                  hold_n=hold, hold_mean=hold_mean, hold_std=hold_std,
                  hold_max=hold_max, self_mean=s_mean, self_max=s_max,
                  ng_hit=ng_hit, ng_total=ng_total,
                  seconds=time.time() - t0, deployed=deployed, deploy_err=deploy_err)
    with JOB_LOCK:
        JOB["result"] = result
        JOB["state"] = "done"
    _save_cfg(dict(cam=cam, folder=folder, roi=[x1, y1, x2, y2],
                   target=target, ng_dir=ng_dir, deploy_dir=deploy_dir,
                   backbone=backbone))
    log("全部完成，总用时 %.1fs" % (time.time() - t0), "ok")


# ---------------------------------------------------------------------------
# ROI 自动建议
# ---------------------------------------------------------------------------
def suggest_roi(folder, n=15):
    files = list_images(folder)
    if not files:
        raise RuntimeError("目录里没有图片")
    step = max(1, len(files) // n)
    picks = files[::step][:n]
    lefts, rights, tops, bottoms = [], [], [], []
    for name in picks:
        g = np.asarray(PIL.Image.open(os.path.join(folder, name)).convert("L"),
                       dtype=np.float32)
        col = g.mean(axis=0)
        row = g.mean(axis=1)
        # 两个方向要用不同门槛，原因是成像结构不同（实测）：
        #   左右 —— 黑边到料面是陡过渡（7 → 130），门槛取 60% 才能躲开渐变带
        #   上下 —— 没有黑边，只是照度梯度（顶部 81 / 峰值 106），门槛取 45% 才不吃掉料面
        on_c = np.where(col > col.max() * 0.60)[0]
        on_r = np.where(row > row.max() * 0.45)[0]
        if not len(on_c) or not len(on_r):
            continue
        lefts.append(int(on_c[0]))
        rights.append(int(on_c[-1]))
        tops.append(int(on_r[0]))
        bottoms.append(int(on_r[-1]))
    if not lefts:
        raise RuntimeError("自动识别亮区失败，请手动拖框")
    # 取交集再留余量：起点用最大值（避开所有图的黑边），终点用最小值
    margin_x = max(8, int((min(rights) - max(lefts)) * 0.02))
    margin_y = max(8, int((min(bottoms) - max(tops)) * 0.02))
    x1 = max(0, max(lefts) + margin_x)
    y1 = max(0, max(tops) + margin_y)
    x2 = min(int(min(rights)) - margin_x, max(rights))
    y2 = min(int(min(bottoms)) - margin_y, max(bottoms))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("各图亮区位置差异过大（物料跑出视野），请检查取景")
    return dict(roi=[x1, y1, x2, y2], sampled=len(lefts),
                drift_x=int(max(lefts) - min(lefts)),
                drift_y=int(max(tops) - min(tops)))


# ---------------------------------------------------------------------------
# 目录浏览
# ---------------------------------------------------------------------------
def list_drives():
    out = []
    for c in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        p = c + ":/"
        if os.path.exists(p):
            out.append(p)
    for p in ["C:/Users/Administrator/Desktop", "C:/Users/Administrator"]:
        if os.path.exists(p) and p not in out:
            out.insert(0, p)
    return out


def browse(path):
    if not path or not os.path.isdir(path):
        return dict(path="", parent="", dirs=list_drives(), images=[])
    path = os.path.normpath(path).replace("\\", "/")
    dirs = []
    try:
        for name in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, name)):
                dirs.append(name)
    except PermissionError:
        pass
    parent = os.path.dirname(path.rstrip("/")) or ""
    return dict(path=path, parent=parent.replace("\\", "/"), dirs=dirs, images=[])


def guess_cam(folder):
    low = (folder or "").lower()
    for i in range(1, 17):
        if ("相机%d" % i) in low or ("cam%d" % i) in low:
            return i
    return None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "PatchCoreBuilder/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _err(self, msg, code=400):
        self._send(code, dict(error=str(msg)))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "build_tool_ui.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
            return
        if u.path == "/api/ping":
            # 供启动前的实例探测用（GET 才能被 urllib 直接探到）
            self._send(200, dict(app="patchcore-builder", pid=os.getpid()))
            return
        if u.path == "/api/preview":
            q = parse_qs(u.query)
            path = unquote(q.get("path", [""])[0])
            try:
                # 直接把原文件发出去，不缩放、不重编码。
                #
                # 为什么不缩放：ROI 坐标是在**原图坐标系**里裁剪的（build 阶段
                # crop_roi 直接切原图）。预览图一旦缩到 1600 宽，用户在预览上
                # 画出的框就落在 1600 坐标系，拿去裁 1800×1300 的原图会整体偏
                # 12.5% —— 看起来像「图片被裁切了」，实际是坐标对不上。
                # 工业图 1800×1300 JPEG 约 300KB，原样传输没有性能问题。
                size = os.path.getsize(path)
                if size <= 12 * 1024 * 1024:
                    ext = os.path.splitext(path)[1].lower()
                    ctype = {".png": "image/png", ".bmp": "image/bmp",
                             ".tif": "image/tiff", ".tiff": "image/tiff",
                             ".webp": "image/webp"}.get(ext, "image/jpeg")
                    with open(path, "rb") as fh:
                        self._send(200, fh.read(), ctype)
                    return
                # 超大图（>12MB）才降采样，并保证宽高比不变
                im = PIL.Image.open(path).convert("RGB")
                w = int(q.get("w", ["2400"])[0])
                if im.width > w:
                    im = im.resize((w, max(1, round(im.height * w / im.width))),
                                   PIL.Image.BILINEAR)
                import io
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=92)
                self._send(200, buf.getvalue(), "image/jpeg")
            except Exception as e:
                self._err(e, 404)
            return
        self._err("not found", 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            data = {}
        try:
            if u.path == "/api/config":
                cfg = _load_cfg()
                self._send(200, dict(
                    device=str(DEVICE),
                    default_backbone=DEFAULT_BACKBONE,
                    backbones=[dict(name=k, note=v["note"],
                                    ok=os.path.exists(resolve_base(k)))
                               for k, v in BACKBONES.items()],
                    cam_backbones={str(i): cam_backbone(i) for i in range(1, 9)},
                    last=dict(
                        cam=cfg.get("cam", 1),
                        folder=cfg.get("folder", ""),
                        roi=cfg.get("roi"),
                        target=cfg.get("target", 9000),
                        ng_dir=cfg.get("ng_dir", ""),
                        deploy_dir=cfg.get("deploy_dir", ""),
                        backbone=cfg.get("backbone", DEFAULT_BACKBONE))))
                return
            if u.path == "/api/browse":
                self._send(200, browse(data.get("path") or ""))
                return
            if u.path == "/api/scan":
                folder = (data.get("folder") or "").replace("\\", "/").rstrip("/")
                if not os.path.isdir(folder):
                    raise RuntimeError("目录不存在: " + folder)
                files = list_images(folder)
                subdirs = [d for d in sorted(os.listdir(folder))
                           if os.path.isdir(os.path.join(folder, d))]
                self._send(200, dict(folder=folder, files=files, count=len(files),
                                     subdirs=subdirs[:20],
                                     suggest_cam=guess_cam(folder)))
                return
            if u.path == "/api/suggest_roi":
                self._send(200, suggest_roi(data.get("folder") or "",
                                            int(data.get("n") or 15)))
                return
            if u.path == "/api/build":
                start_build(int(data.get("cam") or 1), data.get("folder") or "",
                            data.get("roi") or [], data.get("target") or 9000,
                            (data.get("ng_dir") or "").replace("\\", "/"),
                            (data.get("deploy_dir") or "").replace("\\", "/"),
                            data.get("backbone") or DEFAULT_BACKBONE)
                self._send(200, dict(job_id="main"))
                return
            if u.path == "/api/job":
                with JOB_LOCK:
                    logs = JOB["logs"]
                    new = logs[JOB["cursor"]:]
                    JOB["cursor"] = len(logs)
                    self._send(200, dict(
                        state=JOB["state"], stage=JOB["stage"], pct=JOB["pct"],
                        new_logs=new, result=JOB["result"], error=JOB["error"],
                        elapsed=round(time.time() - JOB["t0"], 1) if JOB["t0"] else 0))
                return
            if u.path == "/api/check_dir":
                p = (data.get("path") or "").replace("\\", "/").rstrip("/")
                cam = int(data.get("cam") or 1)
                exists = bool(p) and os.path.isdir(p)
                target, root, err = resolve_deploy_dir(p, cam) if exists else (None, None, "目录不存在")
                writable = False
                if target:
                    probe_dir = target if os.path.isdir(target) else os.path.dirname(target)
                    probe = os.path.join(probe_dir, ".pc_write_test")
                    try:
                        with open(probe, "w") as fh:
                            fh.write("x")
                        os.remove(probe)
                        writable = True
                    except Exception:
                        writable = False
                self._send(200, dict(exists=exists, resolved=target or "",
                                     root=root or "", error=err or "",
                                     writable=writable))
                return
            if u.path == "/api/open_file":
                p = data.get("path") or ""
                if os.path.isfile(p):
                    os.startfile(p)  # noqa: S606
                self._send(200, dict(ok=True))
                return
            if u.path == "/api/open":
                p = data.get("path") or ""
                if os.path.isdir(p):
                    os.startfile(p)  # noqa: S606
                self._send(200, dict(ok=True))
                return
            self._err("not found", 404)
        except Exception as e:
            self._err(e)


def probe_port(host, port, timeout=0.6):
    """返回 'alive'（本工具已在跑）/ 'occupied'（被别的程序占）/ 'free'。

    Windows 上 allow_reuse_address=1 会让两个进程绑同一端口却不报错，
    结果请求随机落到某个实例、配置和任务状态互相打架。所以启动前必须探测。
    """
    import urllib.request
    try:
        with urllib.request.urlopen("http://%s:%d/api/ping" % (host, port),
                                    timeout=timeout) as r:
            if r.status == 200:
                return "alive"
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return "occupied"
    except Exception:
        return "free"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="启动后自动开浏览器")
    args = ap.parse_args()

    url = "http://%s:%d" % (args.host, args.port)
    state = probe_port(args.host, args.port)
    if state == "alive":
        print("已有实例在运行，直接打开浏览器: " + url)
        webbrowser.open(url)
        return
    if state == "occupied":
        print("端口 %d 已被其它程序占用，请换一个端口：--port 8767" % args.port)
        return

    print("=" * 60)
    print("  PatchCore 建库工具")
    print("  设备: %s" % DEVICE)
    print("  默认骨干: %s" % DEFAULT_BACKBONE)
    for _k, _v in BACKBONES.items():
        _p = resolve_base(_k)
        print("    %-14s %-22s %s" % (_k, _v["note"],
                                      "✓" if os.path.exists(_p) else "缺失!"))
    print("  地址: %s" % url)
    print("  关闭本窗口即停止服务")
    print("=" * 60)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("已停止")


if __name__ == "__main__":
    main()
