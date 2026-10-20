"""环境自检脚本 - 验证 PatchCore 工程可在本环境运行。"""
import os
import sys

PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patchcore-inspection")
sys.path.insert(0, os.path.join(PROJ, "src"))
os.chdir(PROJ)

print("=" * 62)
print("PatchCore 环境自检")
print("=" * 62)

# 1. 模块导入
print("\n[1] 导入工程模块")
import patchcore  # noqa: E402
import patchcore.backbones  # noqa: E402
import patchcore.common  # noqa: E402
import patchcore.datasets.mvtec as mvtec  # noqa: E402
import patchcore.metrics  # noqa: E402
import patchcore.patchcore  # noqa: E402
import patchcore.sampler  # noqa: E402
import patchcore.utils  # noqa: E402

print("    OK - 全部 8 个模块导入成功")

# 2. 设备
print("\n[2] 计算设备")
import torch  # noqa: E402

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("    torch     :", torch.__version__)
print("    device    :", device)
if torch.cuda.is_available():
    print("    GPU       :", torch.cuda.get_device_name(0))

# 3. Backbone 构造（用 pretrained=False 避免下载）
print("\n[3] Backbone 构造（离线模式 pretrained=False）")
from torchvision import models  # noqa: E402

bb = models.wide_resnet50_2(weights=None)
bb.name, bb.seed = "wideresnet50", 0
print("    wide_resnet50_2 OK, 参数量 %.1fM" % (sum(p.numel() for p in bb.parameters()) / 1e6))

# 4. 端到端 dummy 前向
print("\n[4] PatchCore 端到端 dummy 前向（image_dim=112）")
dim = 112
pc = patchcore.patchcore.PatchCore(torch.device(device))
pc.load(
    backbone=bb,
    layers_to_extract_from=["layer2", "layer3"],
    device=torch.device(device),
    input_shape=[3, dim, dim],
    pretrain_embed_dimension=256,
    target_embed_dimension=256,
    patchsize=3,
    patchstride=1,
    anomaly_scorer_num_nn=1,
    featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(
        percentage=0.1, device=torch.device(device)
    ),
)
train = torch.utils.data.DataLoader(torch.rand(4, 3, dim, dim), batch_size=2)
pc.fit(train)
scores, masks = pc.predict(torch.rand(2, 3, dim, dim))
print("    fit + predict OK")
print("    scores    :", [round(float(s), 4) for s in scores])
print("    mask shape:", masks[0].shape)

# 5. FAISS 索引读写
print("\n[5] FAISS 索引")
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

nn = patchcore.common.FaissNN(False, 4)
feat = np.random.rand(100, 32).astype("float32")
nn.fit(feat)
d, i = nn.run(1, feat[:5])
print("    fit/search OK, 最近邻距离 shape:", np.array(d).shape)
with tempfile.TemporaryDirectory() as td:
    fp = os.path.join(td, "t.faiss")
    nn.save(fp)
    nn2 = patchcore.common.FaissNN(False, 4)
    nn2.load(fp)
    print("    save/load OK, 文件 %.1f KB" % (os.path.getsize(fp) / 1024))

# 6. 加载真实预训练模型
print("\n[6] 加载真实预训练模型（IM224_WR50 / mvtec_carpet）")
mp = os.path.join(
    PROJ,
    "models/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1/models/mvtec_carpet",
)
if os.path.isdir(mp):
    inst = patchcore.patchcore.PatchCore(torch.device(device))
    bb2 = models.wide_resnet50_2(weights=None)
    import pickle

    with open(os.path.join(mp, "patchcore_params.pkl"), "rb") as f:
        params = pickle.load(f)
    print("    模型参数:", params["backbone.name"], params["layers_to_extract_from"])
    bb2.name, bb2.seed = "wideresnet50", None
    inst.load(**{k: v for k, v in params.items() if k != "backbone.name"}, backbone=bb2,
              device=torch.device(device), nn_method=patchcore.common.FaissNN(False, 4))
    inst.anomaly_scorer.load(mp)
    print("    .faiss 索引加载 OK, ntotal =", inst.anomaly_scorer.nn_method.search_index.ntotal)
    s, m = inst.predict(torch.rand(1, 3, 224, 224))
    print("    真实模型推理 OK, score = %.4f, mask = %s" % (float(s[0]), m[0].shape))
else:
    print("    跳过 - 模型目录不存在")

print("\n" + "=" * 62)
print("自检全部通过")
print("=" * 62)
