"""训练 + 交叉验证：用 test 批次的子集建库，再对全批次打分。

跑两个对照：
  runA: 只把"最干净"的 2 张建库 (A1, F1)
  runB: 把"看起来干净"的 4 张建库 (A1, F1, B1, D1)

看两件事：
  1) 分数排序是否稳定 —— 稳定说明检测器在捕捉真实且一致的差异
  2) 被点名最异常的是不是人眼也认可的物理缺陷
"""
import os
import sys
import json

import numpy as np
import torch
import torchvision.transforms as T
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

import patchcore.common
import patchcore.sampler
import patchcore.patchcore as pcmod
from patchcore import patchcore as pc

from probe_test_folder import IMGS, SRC

W = 320
H = 320
PATCH = 3
PERC = 0.999        # 样本太少，几乎不降采样（sampler 要求严格 (0,1)）
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Loader(torch.utils.data.Dataset):
    """极简 loader。注意：_fill_memory_bank 只接受裸 tensor 或带 'image' 键的 dict。"""

    def __init__(self, paths, resize, imagesize):
        self.paths = paths
        self.tf = T.Compose([
            T.Resize(resize), T.CenterCrop(imagesize),
            T.ToTensor(), T.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(PIL.Image.open(self.paths[i]).convert("RGB"))


def make_model():
    return pc.PatchCore(DEV)


def build_bank(paths, resize, imagesize, percentage):
    """按官方 run_patchcore.py 的装配方式建库。"""
    m = make_model()
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    # backbone 输出 (B,C,1,1)，不需要 fc

    m.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=DEV,
        input_shape=(3, resize, resize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=PATCH,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage, DEV, dimension_to_project_features_to=128),
        anomaly_scorer_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    dl = torch.utils.data.DataLoader(
        Loader(paths, resize, imagesize), batch_size=1, shuffle=False, num_workers=0)
    m.fit(dl)
    return m


def score_all(m, paths, resize, imagesize):
    tf = T.Compose([
        T.Resize(resize), T.CenterCrop(imagesize),
        T.ToTensor(), T.Normalize(mean=MEAN, std=STD),
    ])
    out = {}
    for p in paths:
        t = tf(PIL.Image.open(p).convert("RGB")).unsqueeze(0)
        s, mask = m.predict(t)
        out[os.path.basename(p)] = (float(s[0]), mask[0])
    return out


def run(tag, train_keys, resize, imagesize, percentage):
    all_paths = [os.path.join(SRC, v) for v in IMGS.values()]
    train_paths = [os.path.join(SRC, IMGS[k]) for k in train_keys]

    print(f"\n{'='*74}")
    print(f"[{tag}]  建库 {len(train_paths)} 张 -> {[k for k in train_keys]}")
    print(f"       resize={resize} imagesize={imagesize} pct={percentage}")
    print(f"{'='*74}")

    bank = build_bank(train_paths, resize, imagesize, percentage)
    n = bank.anomaly_scorer.nn_method.search_index.ntotal
    print(f"  记忆库向量数: {n}")

    res = score_all(bank, all_paths, resize, imagesize)

    train_names = {os.path.basename(p) for p in train_paths}
    rows = []
    for name, (s, _) in res.items():
        rows.append((name, s, name in train_names))
    rows.sort(key=lambda x: -x[1])

    print(f"  {'图片':<46} {'score':>9}  建库样本")
    for name, s, is_tr in rows:
        print(f"  {name:<46} {s:9.3f}  {'<-- 是' if is_tr else ''}")

    tr_scores = [s for _, s, t in rows if t]
    oth_scores = [s for _, s, t in rows if not t]
    if tr_scores and oth_scores:
        print(f"\n  建库样本分数: {min(tr_scores):.3f} ~ {max(tr_scores):.3f}")
        print(f"  其他样本分数: {min(oth_scores):.3f} ~ {max(oth_scores):.3f}")
        print(f"  分离度(其他min - 建库max): {min(oth_scores)-max(tr_scores):+.3f}")
    return res, rows


if __name__ == "__main__":
    import io
    ctx = io.StringIO()
    old = sys.stdout
    sys.stdout = ctx
    try:
        # 尺寸必须 >= imagesize 即可；用 224 贴近 MVTec 原生设置
        run("A", ["A1_flatsheet_shelf", "F1_plain_low"], 256, 224, 0.999)
        run("B", ["A1_flatsheet_shelf", "F1_plain_low",
                  "B1_cover_edge", "D1_hand_hold"], 256, 224, 0.999)
    finally:
        sys.stdout = old
    print(ctx.getvalue())

# ---- 追加实验：把"分数最低"的 1e96d5be 当作正常样本加进库 ----
