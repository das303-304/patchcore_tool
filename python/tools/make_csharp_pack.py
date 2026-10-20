# -*- coding: utf-8 -*-
"""生成跨语言对齐基准包：给 C# 测试程序用。

为什么要 PNG 而不是原始 JPG：
  JPG 解码器不同（PIL 用 libjpeg、C# 用 GDI+）会差 ±1 灰阶，
  会把「模型是否一致」和「解码器是否一致」两个问题混在一起。
  PNG 无损，两边读出来必然相同，差异就只剩模型本身。

产物：
  csharp_test/pack/<site>/images/*.png    （24 位 RGB，R=G=B，可直接当灰度用）
  csharp_test/pack/<site>/expected.json   （每张图的期望分数 + 判定 + 阈值）
  csharp_test/pack/<site>/patchcore_e2e.onnx  （复制一份，方便整包拷走）
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnxruntime as ort
import PIL.Image

HERE = os.path.dirname(os.path.abspath(__file__))
OK_ROOT = "C:/Users/Administrator/Desktop/2026-10-08/2026-10-08"
NG_ROOT = "C:/Users/Administrator/Desktop/NG"
PACK = os.path.join(HERE, "csharp_test", "pack")

# 目录名用 ASCII（.bat 里不能有中文，否则 cmd 解析会崩）
CASES = [("皮料相机1", "CAM1", "cam1"), ("皮料相机2", "CAM2", "cam2")]


def main():
    for site, ng_dir, tag in CASES:
        src = os.path.join(HERE, "models_onnx", site)
        with open(os.path.join(src, "meta_v2.json"), encoding="utf-8") as f:
            cal = json.load(f)
        thr = cal["threshold"]

        sess = ort.InferenceSession(os.path.join(src, "patchcore_roi.onnx"),
                                   providers=["CPUExecutionProvider"])

        out_dir = os.path.join(PACK, tag)
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        ok_dir = os.path.join(OK_ROOT, site)
        ok_files = sorted(os.listdir(ok_dir))
        ng_path = os.path.join(NG_ROOT, ng_dir)
        ng_files = [f for f in sorted(os.listdir(ng_path)) if f.endswith(".jpg")]

        # 4 张 OK + 4 张 NG，正负都有才说明问题
        picks = [(os.path.join(ok_dir, f), "OK") for f in ok_files[:4]]
        picks += [(os.path.join(ng_path, f), "NG") for f in ng_files[:4]]

        items = []
        for p, kind in picks:
            g = np.asarray(PIL.Image.open(p).convert("L"), dtype=np.uint8)
            rgb = np.repeat(g[:, :, None], 3, axis=2)
            name = os.path.splitext(os.path.basename(p))[0] + ".png"
            PIL.Image.fromarray(rgb).save(os.path.join(img_dir, name))

            roi = cal["roi"]
            x = g[roi[1]:roi[3], roi[0]:roi[2]][None, None]
            score = float(sess.run(None, {"gray": x})[0].ravel()[0])
            items.append(dict(file=name, truth=kind, score=round(score, 4),
                              verdict="NG" if score >= thr else "OK"))

        shutil.copy2(os.path.join(src, "patchcore_roi.onnx"),
                     os.path.join(out_dir, "patchcore_roi.onnx"))
        shutil.copy2(os.path.join(src, "meta_v2.json"),
                     os.path.join(out_dir, "meta_v2.json"))

        with open(os.path.join(out_dir, "expected.json"), "w", encoding="utf-8") as f:
            json.dump(dict(site=site, threshold=thr,
                           model="patchcore_roi.onnx",
                           roi=cal["roi"],
                           input_note="先用 roi 裁剪整幅图，再把裁剪图按 (1,1,h,w) uint8 送入",
                           items=items), f, ensure_ascii=False, indent=2)

        print(f"[{site}] {len(items)} 张 → {out_dir}")
        for it in items:
            print(f"    {it['file']:<28} {it['truth']:<3} {it['score']:>8.4f}  {it['verdict']}")


if __name__ == "__main__":
    main()
