"""推理速度基准 + coreset 压缩的精度/速度权衡。

回答两个问题：
  1. 用 GPU 推理能快多少？
  2. 减少记忆库向量数能快多少？精度损失多大？

背景：PatchCore 的耗时分成两段
  - 特征提取（backbone forward）—— 吃 GPU 算力
  - FAISS 最近邻检索 —— **随记忆库向量数线性增长**，且跑在 CPU 上
重建后记忆库从 10897 涨到 29164 向量，检索段必然变慢。
"""
import os
import sys
import time
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
ROI = {"皮料相机1": (540, 30, 1780, 1270), "皮料相机2": (15, 30, 1400, 1270)}
DATA = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"


def make_tf(roi, resize=256, imagesize=224):
    return T.Compose([T.Lambda(lambda im: im.crop(roi)), T.Resize(resize),
                      T.CenterCrop(imagesize), T.ToTensor(),
                      T.Normalize(mean=MEAN, std=STD)])


def bench(det, tf, img_path, n=12, warmup=3):
    t = tf(PIL.Image.open(img_path).convert("RGB")).unsqueeze(0)
    dev = det.device
    with torch.no_grad():
        for _ in range(warmup):
            det.model.predict(t)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        st = time.perf_counter()
        for _ in range(n):
            det.model.predict(t)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        total = (time.perf_counter() - st) / n * 1000

        # 只算特征提取（把张量搬到设备）
        tt = t.to(torch.float).to(dev)
        for _ in range(warmup):
            det.model._embed(tt)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        st = time.perf_counter()
        for _ in range(n):
            det.model._embed(tt)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        emb = (time.perf_counter() - st) / n * 1000
    return total, emb, total - emb


def build_bank_percentage(cam, percentage, device):
    """用指定 coreset 比例重建记忆库（不落盘）。"""
    roi = ROI[cam]
    d = os.path.join(DATA, cam)
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
    paths = [os.path.join(d, f) for f in fs]

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.tf = make_tf(roi)

        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return self.tf(PIL.Image.open(paths[i]).convert("RGB"))

    bb = patchcore.backbones.load("wideresnet50")
    bb.name = "wideresnet50"
    m = pc.PatchCore(device)
    m.load(backbone=bb, layers_to_extract_from=["layer2", "layer3"],
           device=device, input_shape=[3, 224, 224],
           pretrain_embed_dimension=1024, target_embed_dimension=1024,
           patchsize=3, patchstride=1, anomaly_scorer_num_nn=1,
           featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
               percentage, device, dimension_to_project_features_to=128),
           nn_method=patchcore.common.FaissNN(False, 4))
    dl = torch.utils.data.DataLoader(DS(), batch_size=1, shuffle=False, num_workers=0)
    ctx = io.StringIO(); old = sys.stdout; sys.stdout = ctx
    try:
        m.fit(dl)
    finally:
        sys.stdout = old
    return m, paths


if __name__ == "__main__":
    print("=" * 94)
    print("一、当前模型（29164 向量）GPU vs CPU")
    print("=" * 94)

    from infer_custom import LeatherInspector
    img = os.path.join(DATA, "皮料相机1",
                       sorted(os.listdir(os.path.join(DATA, "皮料相机1")))[0])

    for dev_name in ["cuda:0", "cpu"]:
        if dev_name.startswith("cuda") and not torch.cuda.is_available():
            continue
        det = LeatherInspector(os.path.join(HERE, "models", "cam1"),
                              device=torch.device(dev_name), roi=ROI["皮料相机1"])
        tf = make_tf(ROI["皮料相机1"])
        n = det.model.anomaly_scorer.nn_method.search_index.ntotal
        tot, emb, srch = bench(det, tf, img)
        pct = tot / 10
        print(f"{dev_name:8s} 向量{n:6d}  总 {tot:7.1f}ms  "
              f"特征 {emb:6.1f}ms  检索 {srch:7.1f}ms  "
              f"({1000/tot:5.2f} FPS)  占1s节拍 {pct:4.1f}%")

    print()
    print("=" * 94)
    print("二、coreset 压缩：向量数 / 速度 / 精度 三方权衡（相机1，GPU）")
    print("=" * 94)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NG = ["107_20261008_101740", "10_20261008_101306", "147_20261008_101941",
          "148_20261008_101942", "179_20261008_102103", "43_20261008_101441",
          "49_20261008_101504", "72_20261008_101606", "75_20261008_101609"]

    rows = []
    print(f"\n{'pct':>7}{'向量数':>9}{'总耗时':>10}{'FPS':>7}"
          f"{'正常max':>10}{'NG中位':>9}{'NG检出':>9}")
    print("-" * 94)

    for pct in [0.10, 0.05, 0.03, 0.02, 0.01]:
        m, paths = build_bank_percentage("皮料相机1", pct, dev)
        nb = m.anomaly_scorer.nn_method.search_index.ntotal
        tf = make_tf(ROI["皮料相机1"])

        # 速度
        t = tf(PIL.Image.open(img).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            for _ in range(3):
                m.predict(t)
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize()
            st = time.perf_counter()
            for _ in range(10):
                m.predict(t)
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize()
            tot = (time.perf_counter() - st) / 10 * 1000

        # 正常样本分数（抽样 60 张，避免太慢）
        with torch.no_grad():
            ss = []
            for p in paths[::6]:
                ti = tf(PIL.Image.open(p).convert("RGB")).unsqueeze(0)
                s, _ = m.predict(ti)
                ss.append(float(s[0]))
            normal_max = float(np.max(ss))

            # NG 分数
            ng_scores = []
            for f in NG:
                p = f"C:/Users/Administrator/Desktop/NG/CAM1/{f}.jpg"
                if not os.path.exists(p):
                    continue
                ti = tf(PIL.Image.open(p).convert("RGB")).unsqueeze(0)
                s, _ = m.predict(ti)
                ng_scores.append(float(s[0]))
            ng_scores = np.array(ng_scores)

        # 用 normal_max 的 1.15 倍当阈值，看能检出几张
        thr = normal_max * 1.15
        det_n = int((ng_scores >= thr).sum())

        print(f"{pct:>7.2f}{nb:>9d}{tot:>9.1f}ms{1000/tot:>7.2f}"
              f"{normal_max:>10.3f}{np.median(ng_scores):>9.3f}"
              f"{det_n:>5}/{len(ng_scores)}")
        rows.append(dict(pct=pct, n_bank=nb, ms=round(tot, 1),
                         normal_max=round(normal_max, 3),
                         ng_median=round(float(np.median(ng_scores)), 3),
                         ng_detected=det_n, ng_total=len(ng_scores)))

    print()
    print("说明：这是**在库内自评**的正常分数，偏乐观（图特征在库里）。")
    print("      用于横向比较不同 pct 的相对变化，不作为绝对阈值依据。")

    out = os.path.join(HERE, "testimgs", "speed_tradeoff.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n明细: {out}")
