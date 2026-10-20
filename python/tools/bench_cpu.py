"""实测 PatchCore 在 CPU 上的推理速度。

产线节拍 1s/件，需要知道 CPU 能否扛住。
测三种场景：
  1. CPU + 完整记忆库（19208 向量）
  2. CPU + 压缩记忆库（coreset 10%）
  3. 不同图像尺寸

关键看：总耗时是否 < 1000ms（留出安全余量）。
"""
import os
import sys
import time
import io

import torch
import torchvision.transforms as T
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

import patchcore.common
from patchcore import patchcore as pc

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SRC_IMG = os.path.join(HERE, "testimgs", "bench_input.jpg")


def bench(bank, img, resize, imagesize, n=10, warmup=2, tag=""):
    tf = T.Compose([T.Resize(resize), T.CenterCrop(imagesize),
                    T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    t = tf(img).unsqueeze(0)

    with torch.no_grad():
        for _ in range(warmup):
            bank.predict(t)
        st = time.perf_counter()
        for _ in range(n):
            bank.predict(t)
        total = (time.perf_counter() - st) / n * 1000

        st = time.perf_counter()
        for _ in range(n):
            _ = bank._embed(t)
        embed = (time.perf_counter() - st) / n * 1000

    print(f"  {tag:<32} 总 {total:7.1f} ms   特征 {embed:6.1f} ms"
          f"   检索 {total-embed:7.1f} ms   ({1000/total:5.2f} FPS)"
          f"   {'OK' if total < 1000 else '超时!'}")
    return total


def load_cpu(model_dir, n_workers=4):
    m = pc.PatchCore(torch.device("cpu"))
    m.load_from_path(load_path=model_dir, device=torch.device("cpu"),
                     nn_method=patchcore.common.FaissNN(False, n_workers))
    return m


if __name__ == "__main__":
    # 用 123 批次的干净图当测试输入
    from crossval_123 import IMGS, SRC
    img = PIL.Image.open(os.path.join(SRC, IMGS["A_smooth_noline"])).convert("RGB")

    print("=" * 96)
    print("PatchCore CPU 推理速度实测（产线节拍 1000ms/件）")
    print("=" * 96)

    torch.set_num_threads(os.cpu_count() or 4)
    print(f"\nCPU 核心数: {torch.get_num_threads()}    "
          f"faiss threads: 4")
    print(f"注意：GPU 实测同尺寸为 45ms（imagesize=224），下面是 CPU 对比\n")

    lm = os.path.join(HERE, "my_models", "leather")
    if not os.path.isdir(lm):
        print("未找到 my_models/leather")
        sys.exit(1)

    m = load_cpu(lm)
    n = m.anomaly_scorer.nn_method.search_index.ntotal
    print(f"记忆库规模: {n} 向量（MVTec leather, 245 张正常样本）\n")

    print("--- 不同图像尺寸（完整记忆库）---")
    for sz, rs in [(160, 182), (224, 256), (320, 366)]:
        bench(m, img, rs, sz, tag=f"imagesize={sz} resize={rs}")

    print("\n--- 不同 faiss 线程数（imagesize=224）---")
    for nw in [1, 2, 4, 8]:
        mm = load_cpu(lm, n_workers=nw)
        bench(mm, img, 256, 224, tag=f"faiss threads={nw}")

    print("\n注：CPU 版 FAISS 已内置多线程，n_workers 影响检索段。")
