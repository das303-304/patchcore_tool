# -*- coding: utf-8 -*-
"""三后端横向对比：torch-GPU / torch-CPU / onnx-CPU。

目的：回答「工控机上到底需不需要 ONNX」。
关键不是本机 GPU 有多快，而是 torch 退到 CPU 之后的表现。

用法：
    python bench_backends.py                # 默认相机1
    python bench_backends.py 皮料相机2
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"

SITES = {
    "皮料相机1": dict(
        model=os.path.join(HERE, "models", "cam1"),
        onnx=os.path.join(HERE, "models_onnx", "皮料相机1"),
        roi=(540, 30, 1780, 1270),
    ),
    "皮料相机2": dict(
        model=os.path.join(HERE, "models", "cam2"),
        onnx=os.path.join(HERE, "models_onnx", "皮料相机2"),
        roi=(15, 30, 1400, 1270),
    ),
}

REPEAT = 10


def _median(xs):
    return float(np.median(xs))


def bench_torch(site, device):
    """返回 (推理中位ms, 特征中位ms, 检索中位ms, score, n_bank)。"""
    import torch
    from infer_custom import LeatherInspector

    cfg = SITES[site]
    det = LeatherInspector(cfg["model"], device=torch.device(device), roi=cfg["roi"])
    cfg_img = cfg

    d = os.path.join(OK_ROOT, site)
    p = os.path.join(d, sorted(os.listdir(d))[0])
    img = PIL.Image.open(p).convert("RGB")
    t = det.transform(img).unsqueeze(0).to(det.device)

    cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for _ in range(3):
            det.model._embed(t)
        if cuda:
            torch.cuda.synchronize()

        t_emb, t_ret = [], []
        emb = None
        for _ in range(REPEAT):
            s = time.perf_counter()
            emb = det.model._embed(t)
            if cuda:
                torch.cuda.synchronize()
            t_emb.append((time.perf_counter() - s) * 1000)

            arr = np.asarray(emb)
            s = time.perf_counter()
            # 返回 (anomaly_scores, query_distances, query_nns) 三个值
            ps, _dist, _nns = det.model.anomaly_scorer.predict([arr])
            t_ret.append((time.perf_counter() - s) * 1000)

    # 与生产链路 det.predict() 的分数对齐核对
    ref_score, _ = det.predict(p)
    score = float(np.max(np.asarray(ps)))
    assert abs(score - float(ref_score)) < 1e-3, f"分数口径不一致 {score} vs {ref_score}"
    n_bank = det.model.anomaly_scorer.nn_method.search_index.ntotal
    total = _median(t_emb) + _median(t_ret)
    return total, _median(t_emb), _median(t_ret), score, n_bank, cfg_img


def bench_onnx(site):
    from onnx_infer import OnnxPatchCore

    cfg = SITES[site]
    det = OnnxPatchCore(cfg["onnx"], providers=["CPUExecutionProvider"])
    d = os.path.join(OK_ROOT, site)
    p = os.path.join(d, sorted(os.listdir(d))[0])
    x = det.preprocess(PIL.Image.open(p).convert("RGB"))

    det.sess.run(None, {det.in_name: x})          # warmup

    t_feat, t_knn = [], []
    emb = None
    for _ in range(REPEAT):
        s = time.perf_counter()
        emb = det.sess.run(None, {det.in_name: x})[0]
        t_feat.append((time.perf_counter() - s) * 1000)

        if emb.ndim == 3:
            emb = emb[0]
        s = time.perf_counter()
        ps = det._patch_scores(np.ascontiguousarray(emb))
        t_knn.append((time.perf_counter() - s) * 1000)

    score = float(np.max(ps))
    n_bank = det.bank.shape[0]
    total = _median(t_feat) + _median(t_knn)
    return total, _median(t_feat), _median(t_knn), score, n_bank


def dir_size_mb(path):
    if not os.path.isdir(path):
        return None
    tot = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1024 / 1024


def main():
    site = sys.argv[1] if len(sys.argv) > 1 else "皮料相机1"

    import torch
    has_cuda = torch.cuda.is_available()

    rows = []
    if has_cuda:
        rows.append(("torch · GPU", *bench_torch(site, "cuda:0")[:5]))
    rows.append(("torch · CPU", *bench_torch(site, "cpu")[:5]))
    rows.append(("onnx  · CPU", *bench_onnx(site)))

    print()
    print("=" * 82)
    print(f"  后端横向对比  [{site}]  记忆库 {rows[0][5]} 向量  重复 {REPEAT} 次取中位")
    print("=" * 82)
    print(f"  {'后端':<14}{'特征提取':>11}{'检索':>11}{'推理合计':>11}"
          f"{'FPS':>8}{'分数':>10}")
    print("  " + "-" * 78)
    for name, tot, fe, rt, sc, nb in rows:
        print(f"  {name:<14}{fe:10.1f}ms{rt:10.1f}ms{tot:10.1f}ms"
              f"{1000 / tot:8.1f}{sc:10.4f}")
    print("=" * 82)

    # 部署体积
    sp = os.path.join(HERE, ".venv", "Lib", "site-packages")
    pkgs = [
        ("torch", os.path.join(sp, "torch")),
        ("onnxruntime", os.path.join(sp, "onnxruntime")),
        ("cv2", os.path.join(sp, "cv2")),
    ]
    print("\n  依赖体积（决定工控机上要装多少东西）")
    print("  " + "-" * 78)
    for name, path in pkgs:
        mb = dir_size_mb(path)
        print(f"  {name:<14}{mb:8.0f} MB" if mb is not None else f"  {name:<14}  未安装")
    onnx_dir = SITES[site]["onnx"]
    if os.path.isdir(onnx_dir):
        print(f"  {'ONNX 模型':<14}{dir_size_mb(onnx_dir):8.0f} MB")
    torch_model = SITES[site]["model"]
    if os.path.isdir(torch_model):
        print(f"  {'torch 模型':<14}{dir_size_mb(torch_model):8.0f} MB")
    print()


if __name__ == "__main__":
    main()
