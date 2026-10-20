"""用自己的皮料（正常样本）建 PatchCore 记忆库 —— 完全无训练。

这不是"训练"，没有任何权重更新、没有反向传播、没有 loss。
所谓 fit 只是把正常样本过一遍冻结的 ImageNet 预训练网络，
抽出 patch 特征 → coreset 子采样 → 存成一个 FAISS 近邻索引文件。
产物是 memory bank，可以直接拷贝到产线机器上加载使用。

用法:
    # 建记忆库（只需正常样本）
    python train_custom.py --data <你的皮料目录> --classname leather --out ../my_models/leather

    # 用自建记忆库做推理测试
    python train_custom.py --data <你的皮料目录> --classname leather --out ../my_models/leather --test

目录要求:
    <你的皮料目录>/leather/good/*.png     <-- 只要这个就能建库
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, "patchcore-inspection")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)

import patchcore.common  # noqa: E402
import patchcore.sampler  # noqa: E402
from custom_dataset import CustomDataset  # noqa: E402
from patchcore import patchcore as pc  # noqa: E402

# 沿用官方最优配置（IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1）
CFG = dict(
    backbone="wideresnet50",
    layers=["layer2", "layer3"],
    pretrain_dim=1024,
    target_dim=1024,
    patchsize=3,
    num_nn=1,
    percentage=0.1,      # coreset 保留比例；样本少时可调到 0.25~1.0
)


def build(device, args):
    from torchvision import models

    bb = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    bb.name, bb.seed = CFG["backbone"], 0

    model = pc.PatchCore(device)
    model.load(
        backbone=bb,
        layers_to_extract_from=CFG["layers"],
        device=device,
        input_shape=[3, args.imagesize, args.imagesize],
        pretrain_embed_dimension=CFG["pretrain_dim"],
        target_embed_dimension=CFG["target_dim"],
        patchsize=CFG["patchsize"],
        patchstride=1,
        anomaly_scorer_num_nn=CFG["num_nn"],
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=args.percentage, device=device
        ),
        nn_method=patchcore.common.FaissNN(False, 4),  # CPU faiss，别开 on_gpu
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="你的皮料根目录")
    ap.add_argument("--classname", default="leather")
    ap.add_argument("--out", required=True, help="记忆库输出目录")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--imagesize", type=int, default=224)
    ap.add_argument("--percentage", type=float, default=CFG["percentage"],
                    help="coreset 保留比例。样本<50张建议 0.25~1.0")
    ap.add_argument("--augment", action="store_true", help="训练时做轻微增强")
    ap.add_argument("--test", action="store_true", help="建库后用测试集验证")
    ap.add_argument("--roi", default=None,
                    help="裁掉背景的 ROI，格式 x0,y0,x1,y1（原图像素坐标）。"
                         "强烈建议指定：黑边/背景是最大误报源")
    args = ap.parse_args()

    roi = None
    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
        if len(roi) != 4:
            ap.error("--roi 需要 4 个整数: x0,y0,x1,y1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    # ---- 数据集 ----
    train_ds = CustomDataset(
        args.data, args.classname, resize=args.resize, imagesize=args.imagesize,
        mode="train", augment=args.augment, roi=roi,
    )
    print(f"\n训练集（正常样本）: {len(train_ds)} 张")
    if roi:
        print(f"  ROI: {roi}  (已裁掉背景)")
    print(f"  分布: {train_ds.stats()}")
    if len(train_ds) < 20:
        print("  ⚠️  正常样本少于 20 张，记忆库会过于稀疏，误报率可能偏高")

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=False, num_workers=0
    )

    # ---- 建库 ----
    model = build(device, args)
    print(f"\n开始构建记忆库（coreset percentage={args.percentage}）...")
    t0 = time.time()
    model.fit(train_dl)
    dt = time.time() - t0
    n_bank = model.anomaly_scorer.nn_method.search_index.ntotal
    print(f"完成，耗时 {dt:.1f}s")
    print(f"记忆库规模: {n_bank} 个特征向量 (维度 {CFG['target_dim']})")

    # ---- 保存 ----
    os.makedirs(args.out, exist_ok=True)
    model.save_to_path(args.out)
    faiss_f = os.path.join(args.out, "nnscorer_search_index.faiss")
    pkl_f = os.path.join(args.out, "patchcore_params.pkl")
    print(f"\n已保存:")
    print(f"  {faiss_f}  ({os.path.getsize(faiss_f)/1024/1024:.1f} MB)")
    print(f"  {pkl_f}")

    meta = dict(
        classname=args.classname, n_train=len(train_ds), n_bank=n_bank,
        resize=args.resize, imagesize=args.imagesize,
        percentage=args.percentage, cfg=CFG, build_seconds=round(dt, 1),
    )
    with open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  {os.path.join(args.out, 'train_meta.json')}")

    # ---- 可选：测试集验证 ----
    if args.test:
        test_ds = CustomDataset(
            args.data, args.classname, resize=args.resize,
            imagesize=args.imagesize, mode="test", roi=roi,
        )
        dist = test_ds.stats()
        print(f"\n测试集: {len(test_ds)} 张, 分布: {dist}")
        if set(dist) == {"good"}:
            print("  ⚠️  只有正常样本，无法算 AUROC。只做推理冒烟测试。")

        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=0
        )
        scores, masks, labels, masks_gt = model.predict(test_dl)

        if len(set(labels)) > 1:
            import patchcore.metrics as M

            img_auroc = M.compute_imagewise_retrieval_metrics(scores, labels)["auroc"]
            px = M.compute_pixelwise_retrieval_metrics(masks, masks_gt)
            print(f"\n图像级 AUROC : {img_auroc:.4f}")
            print(f"像素级 AUROC : {px['auroc']:.4f}")
        else:
            print(f"\n异常分数范围: {min(scores):.2f} ~ {max(scores):.2f}")

        # 分数分布对比（判断阈值）
        g = [s for s, l in zip(scores, labels) if l == 0]
        a = [s for s, l in zip(scores, labels) if l == 1]
        if g:
            print(f"正常样本分数: 均值 {np.mean(g):.2f}, 最大 {np.max(g):.2f} (n={len(g)})")
        if a:
            print(f"缺陷样本分数: 均值 {np.mean(a):.2f}, 最小 {np.min(a):.2f} (n={len(a)})")
        if g and a:
            print(f"\n建议阈值（正常最大值与缺陷最小值之间）: "
                  f"{(np.max(g)+np.min(a))/2:.2f}")


if __name__ == "__main__":
    main()
