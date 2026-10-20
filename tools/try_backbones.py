# -*- coding: utf-8 -*-
"""换骨干离线对比实验：检出能力 + 特征耗时。

只做筛选，不改现场任何模型。流程与 build_tool 一致：
    ROI 裁剪 -> Resize256 -> CenterCrop224 -> 归一化 -> backbone -> patch 特征
    -> greedy coreset -> kNN 分数

评估口径（与建库工具相同）：
    阈值 = max(holdout_mean + 6*holdout_std, holdout_max * 1.02)
    holdout = 未参与建库的 OK 图（避免自评偏乐观）

用法:
    python _archive/try_backbones.py                       # 默认 4 个骨干
    python _archive/try_backbones.py --backbones resnet50 wideresnet50
    python _archive/try_backbones.py --n-ok 150 --n-hold 80 --target 2000
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import PIL.Image
import torch
import torchvision.models as models
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "patchcore-inspection", "src"))

import patchcore.common  # noqa: E402
from patchcore import patchcore as pc  # noqa: E402
from build_tool import greedy_coreset, knn_scores  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
OK_DIR = os.path.join(ROOT, "ok_data", "皮料相机1", "good")
NG_DIR = r"C:/Users/Administrator/Desktop/NG/CAM1"

# 骨干候选：宽残差50（现状） / 标准50（维度相同） / 34 / 18（最小）
CANDIDATES = {
    "wideresnet50": lambda: models.wide_resnet50_2(weights="IMAGENET1K_V1"),
    "resnet50":     lambda: models.resnet50(weights="IMAGENET1K_V1"),
    "resnet34":     lambda: models.resnet34(weights="IMAGENET1K_V1"),
    "resnet18":     lambda: models.resnet18(weights="IMAGENET1K_V1"),
}

TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_cropped(path, roi):
    g = PIL.Image.open(path).convert("RGB")
    return g.crop((roi[0], roi[1], roi[2], roi[3]))


def build_core(backbone, device, embed_dim=1024):
    core = pc.PatchCore(device)
    core.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=[3, 224, 224],
        pretrain_embed_dimension=embed_dim,
        target_embed_dimension=embed_dim,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    return core.eval()


def extract(core, files, roi, device):
    """批量提特征，返回 (n, 784, D) numpy 与单帧耗时。"""
    out, t_total = [], 0.0
    for fp in files:
        x = TF(load_cropped(fp, roi)).unsqueeze(0).to(device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            f = core._embed(x, detach=False)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_total += time.perf_counter() - t0
        out.append(f.detach().float().cpu().numpy())
    return np.stack(out, 0), t_total / max(len(files), 1) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--n-ok", type=int, default=150, help="参与建库的 OK 图数")
    ap.add_argument("--n-hold", type=int, default=80, help="留出评估的 OK 图数")
    ap.add_argument("--target", type=int, default=2000, help="coreset 向量数")
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(ROOT, "models_onnx", "cam1", "meta_v2.json"),
                          encoding="utf-8"))
    roi = meta["roi"]
    thr_now = float(meta["threshold"])

    all_ok = sorted(glob.glob(os.path.join(OK_DIR, "*.jpg")))
    all_ng = sorted(glob.glob(os.path.join(NG_DIR, "*.jpg")))
    need = args.n_ok + args.n_hold
    if len(all_ok) < need:
        raise SystemExit("OK 图不足：需要 %d，实际 %d" % (need, len(all_ok)))

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(all_ok))
    train_files = [all_ok[i] for i in np.sort(perm[:args.n_ok])]
    hold_files = [all_ok[i] for i in np.sort(perm[args.n_ok:need])]

    print("=" * 78)
    print("  换骨干离线对比（cam1, ROI=%s）" % roi)
    print("=" * 78)
    print("  建库 %d 张 OK / 留出 %d 张 OK / NG %d 张 / coreset %d / dim %d"
          % (len(train_files), len(hold_files), len(all_ng), args.target, args.embed_dim))
    print("  参考：现状 wideresnet50 现场阈值 = %.2f，NG 检出 9/12" % thr_now)

    rows = []
    for name in args.backbones:
        if name not in CANDIDATES:
            print("\n[%s] 未知骨干，跳过" % name)
            continue
        print("\n[%s]" % name, flush=True)
        try:
            t0 = time.time()
            backbone = CANDIDATES[name]().to(device).eval()
            core = build_core(backbone, device, args.embed_dim)

            tr, ms_tr = extract(core, train_files, roi, device)
            ho, ms_ho = extract(core, hold_files, roi, device)
            ng, ms_ng = extract(core, all_ng, roi, device)
            print("    特征 %s  patch=%s  单帧 %.1f ms"
                  % (tr.shape, tr.shape[1], (ms_tr + ms_ho + ms_ng) / 3))

            t1 = time.time()
            bank = greedy_coreset(tr.reshape(-1, tr.shape[-1]), args.target)
            print("    记忆库 %s  coreset 用时 %.0fs"
                  % (bank.shape, time.time() - t1), flush=True)

            ho_s = knn_scores(ho, bank)
            ng_s = knn_scores(ng, bank)
            tr_s = knn_scores(tr, bank)

            h_mean, h_std, h_max = float(ho_s.mean()), float(ho_s.std()), float(ho_s.max())
            thr = round(max(h_mean + 6 * h_std, h_max * 1.02), 2)
            hit = int((ng_s >= thr).sum())
            margin = (float(ng_s.min()) / h_max) if h_max > 0 else 0.0

            print("    自评(入库) max=%.4f | 留出 mean=%.4f std=%.4f max=%.4f"
                  % (float(tr_s.max()), h_mean, h_std, h_max))
            print("    建议阈值 %.2f  NG 检出 %d/%d  NG_min=%.4f  NG_max=%.4f"
                  % (thr, hit, len(ng_s), float(ng_s.min()), float(ng_s.max())))
            print("    分离度 NG_min/OK_max = %.2f  (>1 表示可完全分开)" % margin)

            rows.append(dict(
                backbone=name, patch=int(tr.shape[1]), dim=int(tr.shape[-1]),
                bank=int(bank.shape[0]),
                embed_ms=round((ms_tr + ms_ho + ms_ng) / 3, 1),
                self_max=round(float(tr_s.max()), 4),
                hold_mean=round(h_mean, 4), hold_std=round(h_std, 4),
                hold_max=round(h_max, 4), thr=thr,
                ng_hit=hit, ng_total=len(ng_s),
                ng_min=round(float(ng_s.min()), 4), ng_max=round(float(ng_s.max()), 4),
                ng_scores=[round(float(v), 3) for v in ng_s],
                margin=round(margin, 2), seconds=round(time.time() - t0, 1),
            ))
            del core, backbone, tr, ho, ng, bank
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("    [失败] %s: %s" % (type(e).__name__, e))

    print("\n" + "=" * 78)
    print("  汇总")
    print("=" * 78)
    print("  %-14s %6s %5s %6s %9s %9s %7s %8s"
          % ("backbone", "patch", "dim", "embed", "OK_max", "thr", "NG检出", "分离度"))
    for r in rows:
        print("  %-14s %6d %5d %5.0fms %9.4f %9.2f %5d/%-3d %8.2f"
              % (r["backbone"], r["patch"], r["dim"], r["embed_ms"],
                 r["hold_max"], r["thr"], r["ng_hit"], r["ng_total"], r["margin"]))
    print("\n  NG 分数明细：")
    for r in rows:
        print("    %-14s %s" % (r["backbone"], r["ng_scores"]))

    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\n  明细 -> %s" % args.out)


if __name__ == "__main__":
    main()
