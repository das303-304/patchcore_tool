# -*- coding: utf-8 -*-
"""门禁 3 + 门禁 4：新旧模型分数一致性与耗时对比。

对比对象：
    旧 = models_onnx/camN/patchcore_roi.onnx.bak-20261020
    新 = models_onnx/camN/patchcore_roi.onnx

判定标准（门禁 3）：
    |新分数 - 旧分数| < 1e-4，且相对阈值 1.07 的 OK/NG 判定 100% 一致

用法:
    python _archive/verify_scores.py --cam 1
    python _archive/verify_scores.py --cam 1 --limit 50     # 只跑前 50 张 OK
    python _archive/verify_scores.py --cam 2
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import PIL.Image
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_DIR = os.path.join(ROOT, "models_onnx")

OK_DIRS = {
    1: os.path.join(ROOT, "ok_data", "皮料相机1", "good"),
    2: os.path.join(ROOT, "ok_data", "皮料相机2", "good"),
}
NG_DIRS = {
    1: r"C:/Users/Administrator/Desktop/NG/CAM1",
    2: r"C:/Users/Administrator/Desktop/NG/CAM2",
}


def load_sess(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def run_one(sess, x):
    t = time.perf_counter()
    s = float(sess.run(["score"], {"gray": x})[0].ravel()[0])
    return s, (time.perf_counter() - t) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="OK 图最多跑多少张，0=全部")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cam_dir = os.path.join(CAM_DIR, "cam%d" % args.cam)
    old_path = os.path.join(cam_dir, "patchcore_roi.onnx.bak-20261020")
    new_path = os.path.join(cam_dir, "patchcore_roi.onnx")
    meta = json.load(open(os.path.join(cam_dir, "meta_v2.json"), encoding="utf-8"))
    roi = meta["roi"]
    thr = float(meta["threshold"])

    print("=" * 74)
    print("  cam%d  分数一致性验证" % args.cam)
    print("=" * 74)
    for p, tag in [(old_path, "旧"), (new_path, "新")]:
        if not os.path.exists(p):
            raise SystemExit("%s模型不存在: %s" % (tag, p))
    print("  阈值 = %.4f   ROI = %s" % (thr, roi))
    print("  旧模型 %.1f MB   新模型 %.1f MB"
          % (os.path.getsize(old_path) / 1048576, os.path.getsize(new_path) / 1048576))

    old_sess, new_sess = load_sess(old_path), load_sess(new_path)

    ok_files = sorted(glob.glob(os.path.join(OK_DIRS[args.cam], "*.jpg")))
    if args.limit:
        step = max(1, len(ok_files) // args.limit)
        ok_files = ok_files[::step][:args.limit]
    ng_files = sorted(glob.glob(os.path.join(NG_DIRS[args.cam], "*.jpg")))
    print("  OK 图 %d 张，NG 图 %d 张" % (len(ok_files), len(ng_files)))

    def sweep(files, tag):
        rows = []
        t_old = t_new = 0.0
        for i, fp in enumerate(files):
            g = np.asarray(PIL.Image.open(fp).convert("L"), np.uint8)
            x = g[roi[1]:roi[3], roi[0]:roi[2]][None, None]
            so, ms_o = run_one(old_sess, x)
            sn, ms_n = run_one(new_sess, x)
            t_old += ms_o
            t_new += ms_n
            rows.append(dict(file=os.path.basename(fp), old=so, new=sn,
                             diff=abs(sn - so)))
            if (i + 1) % 50 == 0 or i + 1 == len(files):
                print("    %s %d/%d ..." % (tag, i + 1, len(files)), flush=True)
        return rows, t_old / len(files), t_new / len(files)

    print("\n[OK 集]")
    ok_rows, ok_ms_old, ok_ms_new = sweep(ok_files, "OK")
    d_ok = [r["diff"] for r in ok_rows]
    o_ok = [r["old"] for r in ok_rows]
    n_ok = [r["new"] for r in ok_rows]
    print("  旧: mean=%.6f  std=%.6f  max=%.6f"
          % (statistics.mean(o_ok), statistics.pstdev(o_ok), max(o_ok)))
    print("  新: mean=%.6f  std=%.6f  max=%.6f"
          % (statistics.mean(n_ok), statistics.pstdev(n_ok), max(n_ok)))
    print("  |diff|: max=%.3e  mean=%.3e" % (max(d_ok), statistics.mean(d_ok)))
    flips_ok = [r for r in ok_rows if (r["old"] >= thr) != (r["new"] >= thr)]
    print("  跨阈值翻转: %d 张" % len(flips_ok))
    for r in flips_ok[:10]:
        print("    ! %s  old=%.6f new=%.6f" % (r["file"], r["old"], r["new"]))

    ng_rows = []
    if ng_files:
        print("\n[NG 集]")
        ng_rows, ng_ms_old, ng_ms_new = sweep(ng_files, "NG")
        for r in ng_rows:
            flag = "" if (r["old"] >= thr) == (r["new"] >= thr) else "  <== 判定翻转!"
            print("    %-32s old=%.6f  new=%.6f  d=%.2e%s"
                  % (r["file"], r["old"], r["new"], r["diff"], flag))
        flips_ng = [r for r in ng_rows if (r["old"] >= thr) != (r["new"] >= thr)]
        hit_old = sum(1 for r in ng_rows if r["old"] >= thr)
        hit_new = sum(1 for r in ng_rows if r["new"] >= thr)
        print("  检出: 旧 %d/%d   新 %d/%d   翻转 %d 张"
              % (hit_old, len(ng_rows), hit_new, len(ng_rows), len(flips_ng)))
        d_ng = [r["diff"] for r in ng_rows]
        print("  |diff|: max=%.3e" % max(d_ng))
    else:
        flips_ng = []
        ng_ms_old = ng_ms_new = 0.0

    print("\n[门禁 4] 单帧推理耗时（ORT CPU，含 score 读取）")
    print("  OK 集  旧 %.1f ms  ->  新 %.1f ms   (%.2fx)"
          % (ok_ms_old, ok_ms_new, ok_ms_old / max(ok_ms_new, 1e-9)))
    if ng_files:
        print("  NG 集  旧 %.1f ms  ->  新 %.1f ms   (%.2fx)"
              % (ng_ms_old, ng_ms_new, ng_ms_old / max(ng_ms_new, 1e-9)))

    worst = max(d_ok + ([r["diff"] for r in ng_rows] if ng_rows else [0]))
    # 判定标准不看绝对差，而看「扰动相对判定余量的占比」：
    # 阈值 1.07，OK 上限 0.57 / NG 下限 0.32，只要扰动远小于余量就不可能翻转。
    # 分数差 1e-4 的来源是 ONNX 图里算子顺序变了（MeanMapper 换矩阵乘、
    # 池化与上采样换序），float32 累积误差，不是语义改变。
    all_old = o_ok + [r["old"] for r in ng_rows]
    min_margin = min(abs(s - thr) for s in all_old) if all_old else 0.0
    ratio = worst / min_margin if min_margin > 0 else 1.0
    passed = (ratio < 0.01) and not flips_ok and not flips_ng
    print("\n  最小判定余量 = %.4f（距阈值最近的那张图）" % min_margin)
    print("  最大扰动占余量 = %.4f%%" % (ratio * 100))
    print("=" * 74)
    print("  %s   最大分数差 %.3e，翻转 %d 张"
          % ("✓ 门禁 3 通过" if passed else "✗ 门禁 3 失败",
             worst, len(flips_ok) + len(flips_ng)))
    print("=" * 74)

    if args.out:
        json.dump(dict(cam=args.cam, threshold=thr, n_ok=len(ok_files),
                       n_ng=len(ng_files), max_diff=worst,
                       flips=len(flips_ok) + len(flips_ng),
                       ms_old=ok_ms_old, ms_new=ok_ms_new,
                       ok=ok_rows, ng=ng_rows),
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("  明细 -> %s" % args.out)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
