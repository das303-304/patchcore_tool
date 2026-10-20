"""进一步确认 NG 图与 OK 同名图是否为「同一张图重编码」。

判据：
  1. 相关系数（>0.999 基本可判定同一图）
  2. 差异 >10 灰阶的像素占比（若是重编码，应极小）
  3. 差异像素的分布形态（重编码差异集中在强边缘，随机分布则是不同图）
"""
import os

import numpy as np
from PIL import Image

BASE_NG = "C:/Users/Administrator/Desktop/NG"
BASE_OK = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
PAIRS = [("CAM1", "皮料相机1"), ("CAM2", "皮料相机2")]


def main():
    print(f"{'文件':<34}{'相关系数':>10}{'>10的占比':>11}{'>30的占比':>11}{'判定':>12}")
    print("-" * 82)
    verdicts = []
    for ncam, ocam in PAIRS:
        dng = os.path.join(BASE_NG, ncam)
        for f in sorted(os.listdir(dng)):
            po = os.path.join(BASE_OK, ocam, f)
            pn = os.path.join(dng, f)
            if not os.path.exists(po):
                continue
            a = np.asarray(Image.open(pn).convert("L")).astype(np.float64)
            b = np.asarray(Image.open(po).convert("L")).astype(np.float64)
            if a.shape != b.shape:
                continue
            r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            d = np.abs(a - b)
            p10 = (d > 10).mean() * 100
            p30 = (d > 30).mean() * 100

            if r > 0.999 and p10 < 0.5:
                v = "同一张图"
            elif r > 0.995:
                v = "极可能同图"
            else:
                v = "不同图"
            verdicts.append(v)
            print(f"{ncam}/{f:<30}{r:10.6f}{p10:10.4f}%{p30:10.4f}%{v:>12}")

    print("-" * 82)
    from collections import Counter
    print("判定统计:", dict(Counter(verdicts)))


if __name__ == "__main__":
    main()
