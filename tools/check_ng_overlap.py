"""核对 NG 图与 OK 批次——是否是同一张图。

用户新交付 `Desktop/NG/`（CAM1 10 张 + CAM2 4 张），
文件名与之前的 OK 批次**全部重名**。必须先确认是不是同一批图，
否则"标签矛盾"会让后续验证结论无意义。
"""
import os

import numpy as np
from PIL import Image

BASE_NG = "C:/Users/Administrator/Desktop/NG"
BASE_OK = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
PAIRS = [("CAM1", "皮料相机1"), ("CAM2", "皮料相机2")]


def main():
    print(f"{'相机/文件':<46}{'像素一致':>10}{'最大差':>9}{'均值差':>10}{'rms':>9}")
    print("-" * 86)
    n_same = n_diff = 0
    for ncam, ocam in PAIRS:
        dng = os.path.join(BASE_NG, ncam)
        for f in sorted(os.listdir(dng)):
            po = os.path.join(BASE_OK, ocam, f)
            pn = os.path.join(dng, f)
            tag = f"{ncam}/{f}"
            if not os.path.exists(po):
                print(f"{tag:<46}  OK批次无同名文件")
                continue
            a = np.asarray(Image.open(pn).convert("L")).astype(np.int16)
            b = np.asarray(Image.open(po).convert("L")).astype(np.int16)
            if a.shape != b.shape:
                print(f"{tag:<46}  尺寸不同 {a.shape} vs {b.shape}")
                n_diff += 1
                continue
            d = np.abs(a - b)
            if d.max() == 0:
                n_same += 1
                print(f"{tag:<46}{'是':>10}{0:>9}{0:>10}{0:>9}")
            else:
                n_diff += 1
                print(f"{tag:<46}{'否':>10}{d.max():>9}{d.mean():>10.3f}"
                      f"{np.sqrt((d.astype(float)**2).mean()):>9.3f}")

    print("-" * 86)
    print(f"完全一致 {n_same} 张，有差异 {n_diff} 张")
    if n_same and not n_diff:
        print("\n>>> 结论：NG 图就是 OK 批次里的同一批文件（仅重存为 RGB）")
        print(">>> 这意味着两类标签互相矛盾，不能直接当缺陷样本用。")
    elif n_same:
        print("\n>>> 结论：部分重名且像素一致，部分不同。需要分别对待。")
    else:
        print("\n>>> 结论：全部不同 —— NG 是独立的新样本，可直接用于验证。")


if __name__ == "__main__":
    main()
