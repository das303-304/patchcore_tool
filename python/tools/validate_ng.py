"""用 NG 图做真正的检出能力验证 —— 之前缺失的那一半。

流程：
  1. 从 OK 训练集中**剔除** 14 张 NG 图（用户修正了标签）
  2. 用剩下的 268 张纯 OK 图重建记忆库
  3. 对全部 282 张评分（268 OK + 14 NG）
  4. 算 AUROC、找最佳阈值、给出各阈值下的漏报/误报

注意：NG 图与 OK 批次的同名文件是同一张图（相关系数 0.9999 已核实），
所以这里用 OK 批次里那份（灰度 L，与训练同源）来保持一致；
NG 文件夹那份是 RGB 重存版，供交叉核对。
"""
import os
import sys
import io
import json

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

import patchcore.common
import patchcore.sampler
from patchcore import patchcore as pc

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"

CAMS = [
    ("皮料相机1", "CAM1", (540, 30, 1780, 1270)),
    ("皮料相机2", "CAM2", (15, 30, 1400, 1270)),
]

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESIZE, IMAGESIZE, PCT = 256, 224, 0.1


class ImgList(torch.utils.data.Dataset):
    def __init__(self, paths, roi):
        self.paths = paths
        self.tf = T.Compose([
            T.Lambda(lambda im: im.crop(roi)),
            T.Resize(RESIZE), T.CenterCrop(IMAGESIZE),
            T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(PIL.Image.open(self.paths[i]).convert("RGB"))


def build_bank(paths, roi):
    dl = torch.utils.data.DataLoader(
        ImgList(paths, roi), batch_size=1, shuffle=False, num_workers=0)
    bb = patchcore.backbones.load("wideresnet50")
    bb.name = "wideresnet50"
    m = pc.PatchCore(DEV)
    m.load(backbone=bb, layers_to_extract_from=["layer2", "layer3"],
           device=DEV, input_shape=[3, IMAGESIZE, IMAGESIZE],
           pretrain_embed_dimension=1024, target_embed_dimension=1024,
           patchsize=3, patchstride=1, anomaly_scorer_num_nn=1,
           featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
               PCT, DEV, dimension_to_project_features_from=128
               if False else None),
           nn_method=patchcore.common.FaissNN(False, 4))
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        m.fit(dl)
    finally:
        sys.stdout = old
    return m


def score(model, paths, roi):
    dl = torch.utils.data.DataLoader(
        ImgList(paths, roi), batch_size=1, shuffle=False, num_workers=0)
    out = {}
    with torch.no_grad():
        for p in paths:
            t = ImgList([p], roi)[0].unsqueeze(0)
            s, _ = model.predict(t)
            out[os.path.basename(p)] = float(s[0])
    return out


def auroc(scores, labels):
    """纯 numpy 实现的 AUROC（避免依赖 sklearn 的版本差异）。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    summary = {}
    for ok_cam, ng_cam, roi in CAMS:
        ok_dir = os.path.join(OK_ROOT, ok_cam)
        ng_dir = os.path.join(NG_ROOT, ng_cam)

        all_ok = [os.path.join(ok_dir, f) for f in sorted(os.listdir(ok_dir))
                  if os.path.splitext(f)[1].lower() in IMG_EXT]
        ng_files = [f for f in sorted(os.listdir(ng_dir))
                    if os.path.splitext(f)[1].lower() in IMG_EXT]
        ng_names = set(ng_files)

        # 关键：从 OK 训练集里剔除用户标为 NG 的那些
        train_ok = [p for p in all_ok if os.path.basename(p) not in ng_names]
        test_ok = [p for p in all_ok if os.path.basename(p) in ng_names]
        test_ng = [os.path.join(ng_dir, f) for f in ng_files]

        print(f"\n{'='*88}")
        print(f"[{ok_cam}]  训练 {len(train_ok)} 张纯 OK  |  "
              f"测试 OK {len(test_ok)} + NG {len(test_ng)}")
        print(f"{'='*88}")
        print(f"  被剔除出训练集的 NG 文件: {sorted(ng_names)}")

        m = build_bank(train_ok, roi)
        print(f"  记忆库向量数: {m.anomaly_scorer.nn_method.search_index.ntotal}")

        sc_train = score(m, train_ok, roi)
        sc_testok = score(m, test_ok, roi)
        sc_ng = score(m, test_ng, roi)

        tr = np.array(list(sc_train.values()))
        to = np.array(list(sc_testok.values())) if sc_testok else np.array([])
        ng = np.array(list(sc_ng.values()))

        print(f"\n  训练OK分数   : 均值 {tr.mean():.3f}  max {tr.max():.3f}  "
              f"p99 {np.percentile(tr,99):.3f}")
        if len(to):
            print(f"  测试OK分数   : {np.array2string(to, precision=3)}")
        print(f"  NG  分数     : " +
              " ".join(f"{v:.3f}" for v in sorted(ng, reverse=True)))

        # --- 检出率 / 误报率（阈值扫描）---
        print(f"\n  {'阈值':>7}{'检出':>7}{'漏报':>7}{'误报(训练OK)':>14}{'误报(测试OK)':>14}")
        print("  " + "-" * 58)
        rows = []
        for thr in [0.70, 0.80, 0.85, 0.90, 0.91, 0.95, 1.00, 1.10, 1.20, 1.50, 2.00]:
            det = int((ng > thr).sum())
            fp_tr = int((tr > thr).sum())
            fp_to = int((to > thr).sum()) if len(to) else 0
            rows.append((thr, det, len(ng) - det, fp_tr, fp_to))
            print(f"  {thr:7.2f}{det:>4}/{len(ng):<3}{len(ng)-det:>7}"
                  f"{fp_tr:>10}/{len(tr):<3}{fp_to:>10}/{len(to):<3}")

        # --- AUROC ---
        all_sc = np.concatenate([tr, ng])
        all_lb = np.concatenate([np.zeros(len(tr)), np.ones(len(ng))])
        auc = auroc(all_sc, all_lb)

        # 最佳阈值：最大化 (检出率 - 误报率) 即 Youden J
        best, bestJ = None, -9
        for thr in np.linspace(0.4, 3.0, 521):
            tpr = (ng > thr).mean()
            fpr = (tr > thr).mean()
            if tpr - fpr > bestJ:
                bestJ, best = tpr - fpr, thr

        print(f"\n  图像级 AUROC = {auc:.4f}")
        print(f"  最佳阈值(Youden) = {best:.3f}  "
              f"检出 {int((ng>best).sum())}/{len(ng)}  "
              f"误报 {int((tr>best).sum())}/{len(tr)}")

        summary[ok_cam] = dict(
            n_train=len(train_ok), n_ng=len(ng),
            n_bank=int(m.anomaly_scorer.nn_method.search_index.ntotal),
            auroc=round(float(auc), 4),
            train_ok_mean=round(float(tr.mean()), 3),
            train_ok_max=round(float(tr.max()), 3),
            ng_min=round(float(ng.min()), 3),
            ng_max=round(float(ng.max()), 3),
            best_thr=round(float(best), 3),
            best_detected=int((ng > best).sum()),
            best_fp=int((tr > best).sum()),
            separation=round(float(ng.min() - tr.max()), 3),
        )

    print(f"\n{'='*88}")
    print("汇总")
    print(f"{'='*88}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open(os.path.join(HERE, "testimgs", "ng_validation.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
