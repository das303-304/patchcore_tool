#!/bin/bash
# PatchCore 环境重建脚本
# 用法: bash setup_env.sh
#
# 说明: 原 requirements.txt 不完整（缺 timm），且未指定 torch 的 CUDA 版本。
#       本脚本按实测可用的版本组合重建环境。

set -e

cd "$(dirname "$0")"

BASE_PY="C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe"
VENV=".venv"

echo "=== 1. 创建虚拟环境 ==="
if [ -d "$VENV" ]; then
    echo "  已存在 $VENV，跳过创建"
else
    "$BASE_PY" -m venv "$VENV"
    echo "  已创建 $VENV"
fi

PY="$VENV/Scripts/python.exe"
"$PY" -m pip install --upgrade pip

echo "=== 2. 安装 CUDA 版 PyTorch (cu124, 适配 RTX 3060 / 驱动 577) ==="
# 注意: 首次会下载约 2.5GB，若中途因删除保护中断可直接重跑（轮子已缓存）
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== 3. 安装工程依赖 ==="
# timm 是 backbones.py 的硬依赖，原 requirements.txt 漏了，必须补上
"$PY" -m pip install \
    "click>=8.0.3" \
    faiss-cpu \
    "matplotlib>=3.5.0" \
    "pillow>=8.4.0" \
    "pretrainedmodels>=0.7.4" \
    "scikit-image>=0.18.3" \
    "scikit-learn>=1.0.1" \
    "scipy>=1.7.1" \
    "tqdm>=4.62.3" \
    timm

echo "=== 4. 安装测试依赖 ==="
"$PY" -m pip install pytest

echo "=== 5. 环境自检 ==="
"$PY" check_env.py

echo ""
echo "=== 完成 ==="
echo "激活环境: source .venv/Scripts/activate"
echo "运行工程: cd patchcore-inspection && PYTHONPATH=src ../.venv/Scripts/python.exe bin/run_patchcore.py --help"
