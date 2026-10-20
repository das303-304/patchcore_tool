"""用 OK 图做留出法交叉验证 —— 检验记忆库质量（误报率）。

关键逻辑：只用 OK 图就够了。
  建库集 = 一部分 OK 图        -> 建记忆库
  留出集 = 剩下的 OK 图        -> 当"未知样本"检测

留出集全部是正常料，所以：
  - 留出集分数全部低   -> 库建得好，无误报
  - 留出集有高分图     -> 那就是误报，必须解决

同时对比「裁ROI」vs「不裁ROI」，量化背景黑边造成的误报。

用法:
    python validate_ok_batch.py                     # 两个相机都跑
    python validate_ok_batch.py --camera 皮料相机1   # 只跑一个
"""
import os
import sys
import io
import json
import argparse

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

DATA_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"

# 料区 ROI（原图像素坐标）。必须留足余量：
# 实测物料边缘会漂移 —— 相机1 起点 444~515，相机2 终点 1419~1520。
# 所以起点取 max+余量、终点取 min-余量，否则边缘黑边会漏进来造成误报。
ROI = {
    "皮料相机1": (540, 30, 1780, 1270),
    "皮料相机2": (15, 30, 1400, 1270),
}

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImgList(torch.utils.data.Dataset):
    """直接吃图片路径列表，可选 ROI 裁剪。"""

    def __init__(self, paths, resize, imagesize, roi=None):
        self.paths = paths
        self.tf = T.Compose(self._build(resize, imagesize, roi))

    @staticmethod
    def _build(resize, imagesize, roi):
        steps = []
        if roi:
            steps.append(T.Lambda(lambda im: im.crop(roi)))
        steps += [T.Resize(resize), T.CenterCrop(imagesize),
                  T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
        return steps

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(PIL.Image.open(self.paths[i]).convert("RGB"))


def score_paths(model, paths, resize, imagesize, roi):
    ds = ImgList(paths, resize, imagesize, roi)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    scores, masks, labels, gts = model.predict(dl)
    return list(scores), masks


def build_bank(paths, resize, imagesize, roi, percentage=0.1):
    ds = ImgList(paths, resize, imagesize, roi)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    bb = patchcore.backbones.load("wideresnet50")
    bb.name = "wideresnet50"

    m = pc.PatchCore(DEV)
    m.load(backbone=bb,
           layers_to_extract_from=["layer2", "layer3"],
           device=DEV,
           input_shape=[3, imagesize, imagesize],
           pretrain_embed_dimension=1024,
           target_embed_dimension=1024,
           patchsize=3, patchstride=1,
           anomaly_scorer_num_nn=1,
           featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
               percentage, DEV, dimension_to_project_features_to=128),
           nn_method=patchcore.common.FaissNN(False, 4))
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        m.fit(dl)
    finally:
        sys.stdout = old
    return m


def holdout_validate(camera, resize=256, imagesize=224, roi=None, pct=0.1,
                     holdout_every=3):
    """双向留出验证。

    注意方法论：不能拿「建库集分数最大值」当阈值 ——
    建库图的特征就在记忆库里，分数被人为压到接近 0，这样比必然误报一片。
    正确做法是只用「留出集分数」估计正常分布，并做双向交叉：
        A 组建库 -> 评 B 组
        B 组建库 -> 评 A 组
    两组结果都是「模型没见过的正常图」，合并后才是无偏的正常分布。
    """
    d = os.path.join(DATA_ROOT, camera)
    fs = sorted(f for f in os.listdir(d)
                if os.path.splitext(f)[1].lower() in IMG_EXT)
    grpA = [os.path.join(d, f) for i, f in enumerate(fs) if i % 2 == 0]
    grpB = [os.path.join(d, f) for i, f in enumerate(fs) if i % 2 == 1]

    tag = "裁ROI" if roi else "不裁ROI"
    print(f"\n{'='*90}")
    print(f"[{camera}] {tag}    A组 {len(grpA)} 张 / B组 {len(grpB)} 张（双向交叉）")
    print(f"{'='*90}")

    hold_all, bank_sizes = [], []
    for bn, (build, hold), lbl in [(1, (grpA, grpB), "A建库->评B"),
                                   (2, (grpB, grpA), "B建库->评A")]:
        m = build_bank(build, resize, imagesize, roi, pct)
        nb = m.anomaly_scorer.nn_method.search_index.ntotal
        bank_sizes.append(nb)
        hs, _ = score_paths(m, hold, resize, imagesize, roi)
        hold_all.extend(hs)
        print(f"  {lbl}: 记忆库 {nb:6d} 向量 -> "
              f"留出 {len(hs)} 张  均值 {np.mean(hs):.3f}  "
              f"最大 {np.max(hs):.3f}  p95 {np.percentile(hs,95):.3f}")

    h = np.array(hold_all)
    mean, std = h.mean(), h.std()
    mx, p99, p95 = h.max(), np.percentile(h, 99), np.percentile(h, 95)

    print(f"\n  【正常分数分布·共 {len(h)} 张留出图】")
    print(f"    均值 {mean:.3f}   标准差 {std:.3f}")
    print(f"    p95 {p95:.3f}   p99 {p99:.3f}   最大 {mx:.3f}")

    print(f"\n  【阈值建议】留出集全为正常料，阈值需高于其最大观测值")
    for k, name in [(0, "max（无余量，最激进）"),
                    (3, "mean+3σ"),
                    (4, "mean+4σ"),
                    (6, "mean+6σ（推荐起点）"),
                    (10, "mean+10σ（保守）")]:
        thr = mx if k == 0 else mean + k * std
        print(f"    {name:<26} = {thr:7.3f}")

    return h, bank_sizes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None, help="只跑某个相机")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--imagesize", type=int, default=224)
    ap.add_argument("--percentage", type=float, default=0.1)
    ap.add_argument("--no-roi", action="store_true", help="不裁 ROI（做对照）")
    args = ap.parse_args()

    cams = [args.camera] if args.camera else list(ROI)
    summary = {}

    for cam in cams:
        roi = None if args.no_roi else ROI[cam]
        h, nb = holdout_validate(cam, args.resize, args.imagesize, roi,
                                 args.percentage)
        summary[cam] = dict(
            n_bank=nb, n_holdout=len(h),
            mean=round(float(h.mean()), 3), std=round(float(h.std()), 3),
            p95=round(float(np.percentile(h, 95)), 3),
            p99=round(float(np.percentile(h, 99)), 3),
            max=round(float(h.max()), 3),
            thr_mean6sigma=round(float(h.mean() + 6 * h.std()), 3),
        )

    print(f"\n{'='*90}")
    print("汇总")
    print(f"{'='*90}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
