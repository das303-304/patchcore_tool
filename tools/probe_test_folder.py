"""对 C:/Users/Administrator/Desktop/test 批次做交叉验证。

目的：判断这批"灰色保护壳"图能否检出缺陷。
方法：多假设交叉验证 —— 用不同子集建库，看分数排序是否稳定、
      以及"被检出最异常的地方"是否落在人眼可见的物理缺陷上。
"""
import os
import sys
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection"))
sys.path.insert(0, os.path.join(HERE, "patchcore-inspection", "src"))

SRC = "C:/Users/Administrator/Desktop/test"
WORK = os.path.join(HERE, "probe_tmp")

# 按内容语义命名（人工观察 10 张图后归纳）
IMGS = {
    "A1_flatsheet_shelf":  "080d03b9559bf43454d70e0ffb49cc7f.jpg",      # 平滑面料特写
    "A2_two_darklines":    "080d03b9559bf43454d70e0ffb49cc7f(1).jpg",  # 两条深色压痕线
    "B1_cover_edge":       "0fd8044ee58abaaada8f5db5e183deba.jpg",      # 壳体折边+背景
    "B2_deepgroove":       "1e96d5bec9d09f66cbba087733fa8e76.jpg",      # 深凹槽
    "C1_dimple_sheet":     "2f90ec8d3923f40b6c72a12f3e5edfed.jpg",      # 可见凹坑
    "C2_dimple_grooves":   "2f90ec8d3923f40b6c72a12f3e5edfed(1).jpg",   # 凹坑+双凹槽
    "D1_hand_hold":        "30868a8726062ae8026762210f6144f2.jpg",      # 手持入镜+背景
    "E1_procase_full":     "5891e1f45ef89a7119c7469f510aab09.jpg",      # 完整 Pro Case
    "F1_plain_low":        "5e87f775ce21de409cea16ea974ed6ea.jpg",      # 平滑低纹理
    "F2_crease_seam":      "5e87f775ce21de409cea16ea974ed6ea(1).jpg",   # 折痕+拼接缝
}


def build(split_name, train_keys):
    """把指定图片拷进 <WORK>/<split>/shell/train/good/ 并建库。"""
    root = os.path.join(WORK, split_name)
    if os.path.exists(root):
        shutil.rmtree(root)
    good = os.path.join(root, "shell", "train", "good")
    os.makedirs(good)
    # 同时建一个空 test 目录（run_patchcore 强制要求）
    os.makedirs(os.path.join(root, "shell", "test", "good"))
    for k in train_keys:
        shutil.copy(os.path.join(SRC, IMGS[k]), os.path.join(good, IMGS[k]))
    return root


if __name__ == "__main__":
    print(__doc__)
    for k, v in IMGS.items():
        p = os.path.join(SRC, v)
        print(f"  {k:22s} {v}  exists={os.path.exists(p)}")
