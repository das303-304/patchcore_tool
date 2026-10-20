"""验证 coreset 压缩比例对精度的影响（双向交叉验证，无偏）。

对每个 pct：A组建库评B组 + B组建库评A组，得到无偏的正常分数分布；
再用同一库给 NG 图打分，看检出能力。

输出：pct / 向量数 / 速度 / 正常分布 / 阈值 / NG 检出
"""
import os
import sys
import time
import io

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
DATA = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_DIR = "C:/Users/Administrator/Desktop/NG"
ROI = {"皮料相机1": (540, 30, 1780, 1270), "皮料相机2": (15, 30, 1400, 1270)}

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, roi):
        self.paths = paths
        self.tf = T.Compose([T.Lambda(lambda im: im.crop(roi)), T.Resize(256),
                             T.CenterCrop(224), T.ToTensor(),
                             T.Normalize(mean=MEAN, std=STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(PIL.Image.open(self.paths[i]).convert("RGB"))


def build(paths, roi, pct):
    bb = patchcore.backbones.load("wideresnet50")
    bb.name = "wideresnet50"
    m = pc.PatchCore(DEV)
    m.load(backbone=bb, layers_to_extract_from=["layer2", "layer3"],
           device=DEV, input_shape=[3, 224, 224],
           pretrain_embed_dimension=1024, target_embed_dimension=1024,
           patchsize=3, patchstride=1, anomaly_scorer_num_nn=1,
           featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
               pct, DEV, dimension_to_project_features_to=128),
           nn_method=patchcore.common.FaissNN(False, 4))
    dl = torch.utils.data.DataLoader(
        DS(paths, roi), batch_size=1, shuffle=False, num_workers=0)
    old = sys.stdout
    sys.stdout = io.StringIO()      # 屏蔽 tqdm 进度条
    try:
        m.fit(dl)
    finally:
        sys.stdout = old
    return m


def scores(m, paths, roi):
    tf = T.Compose([T.Lambda(lambda im: im.crop(roi)), T.Resize(256),
                    T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    out = []
    with torch.no_grad():
        for p in paths:
            t = tf(PIL.Image.open(p).convert("RGB")).unsqueeze(0)
            s, _ = m.predict(t)
            out.append(float(s[0]))
    return np.array(out)


def speed(m, roi, sample):
    tf = T.Compose([T.Lambda(lambda im: im.crop(roi)), T.Resize(256),
                    T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    t = tf(PIL.Image.open(sample).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        for _ in range(3):
            m.predict(t)
        if str(DEV).startswith("cuda"):
            torch.cuda.synchronize()
        st = time.perf_counter()
        for _ in range(8):
            m.predict(t)
        if str(DEV).startswith("cuda"):
            torch.cuda.synchronize()
        return (time.perf_counter() - st) / 8 * 1000


if __name__ == "__main__":
    cam = sys.argv[1] if len(sys.argv) > 1 else "皮料相机1"
    roi = ROI[cam]
    d = os.path.join(DATA, cam)
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
    A = [os.path.join(d, f) for i, f in enumerate(fs) if i % 2 == 0]
    B = [os.path.join(d, f) for i, f in enumerate(fs) if i % 2 == 1]

    ngd = os.path.join(NG_DIR, "CAM1" if cam == "皮料相机1" else "CAM2")
    ng = [os.path.join(ngd, f) for f in sorted(os.listdir(ngd))
          if f.lower().endswith(".jpg")]
    # 排除未修改的 29（相机1）
    ng = [p for p in ng if not os.path.basename(p).startswith("29_")]

    print("=" * 100)
    print(f"[{cam}] coreset 压缩 vs 精度（双向交叉验证，无偏）   NG {len(ng)} 张")
    print("=" * 100)
    print(f"\n{'pct':>6}{'向量数':>9}{'速度':>10}{'正常均值':>10}{'std':>7}"
          f"{'正常max':>9}{'建议阈值':>10}{'NG最小':>9}{'NG检出':>9}")
    print("-" * 100)

    results = []
    for pct in [0.10, 0.05, 0.03, 0.02]:
        # 双向交叉
        hold = []
        m1 = build(A, roi, pct)
        hold.append(scores(m1, B, roi))
        m2 = build(B, roi, pct)
        hold.append(scores(m2, A, roi))
        h = np.concatenate(hold)
        nb = m1.anomaly_scorer.nn_method.search_index.ntotal

        sp = speed(m1, roi, A[0])

        # NG 用两个库各打一遍取平均（更稳）
        ng1 = scores(m1, ng, roi)
        ng2 = scores(m2, ng, roi)
        ngm = (ng1 + ng2) / 2

        mean, std, mx = h.mean(), h.std(), h.max()
        thr = mean + 6 * std

        # 用 thr 判 NG，同时要求正常不误报
        det = int((ngm >= thr).sum())
        print(f"{pct:>6.2f}{nb:>9d}{sp:>9.1f}ms{mean:>10.3f}{std:>7.3f}"
              f"{mx:>9.3f}{thr:>10.3f}{ngm.min():>9.3f}{det:>5}/{len(ng)}")

        results.append(dict(pct=pct, n_bank=nb, ms=round(sp, 1),
                            mean=round(float(mean), 3), std=round(float(std), 3),
                            max=round(float(mx), 3), thr=round(float(thr), 3),
                            ng_min=round(float(ngm.min()), 3),
                            ng_detected=det, ng_total=len(ng)))

        del m1, m2
        if str(DEV).startswith("cuda"):
            torch.cuda.empty_cache()

    print()
    print("说明：正常统计量来自留出集（无偏）；建议阈值 = mean+6σ；")
    print("      NG 分数取两个库的平均，降低 coreset 随机性影响。")

    import json
    out = os.path.join(HERE, "testimgs", f"pct_tradeoff_{cam}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n明细: {out}")
