# -*- coding: utf-8 -*-
"""用当前骨干模板重建某相机的记忆库（口径与 build_tool 完全一致）。

与 build_tool._run_build 的差异：只做建库必需的部分（无 HTTP/进度/部署），
便于换骨干后批量重跑。复用其 build_front / build_final / greedy_coreset /
knn_scores / crop_roi，保证与建库工具算出来的库和阈值同口径。

流程：
    front.onnx（当前 _base/model.onnx 骨干 + 预处理）
    -> 全量 OK 图提特征
    -> 留出标定（80% 建临时库，给未入库图打分-> 定阈值）
    -> 全量 coreset
    -> build_final 拼装 patchcore_roi.onnx
    -> 建库图自评 + 写 meta_v2.json / memory_bank_v2.npy

用法:
    python _archive/rebuild_cam.py --cam 1
    python _archive/rebuild_cam.py --cam 2 --target 3000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from build_tool import (BACKBONES, BASE_MODEL, OUT_ROOT, PATCH_GRID,  # noqa: E402
                        build_final, build_front, crop_roi, greedy_coreset,
                        knn_scores, list_images, resolve_base)


def backbone_name_of(path):
    """模板路径 -> 逻辑骨干名（写进 meta 用，便于界面/部署比对）。"""
    ap = os.path.abspath(path)
    for k in BACKBONES:
        if os.path.abspath(resolve_base(k)) == ap:
            return k
    stem = os.path.basename(path).replace("model.", "").replace(".onnx", "")
    return stem if stem in BACKBONES else stem

OK_DIRS = {
    1: os.path.join(ROOT, "ok_data", "皮料相机1", "good"),
    2: os.path.join(ROOT, "ok_data", "皮料相机2", "good"),
}
NG_DIRS = {
    1: r"C:/Users/Administrator/Desktop/NG/CAM1",
    2: r"C:/Users/Administrator/Desktop/NG/CAM2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--ok-dir", default="")
    ap.add_argument("--ng-dir", default="")
    ap.add_argument("--roi", default="", help="x1,y1,x2,y2；留空则沿用已有 meta_v2.json")
    ap.add_argument("--target", type=int, default=6000)
    ap.add_argument("--base", default="",
                    help="骨干模板路径；留空用 models_onnx/_base/model.onnx（当前为 resnet50）。"
                         "cam2 需指定 model.wresnet50.onnx —— 它换成小骨干会丢失 NG 判别力")
    ap.add_argument("--out-dir", default="",
                    help="输出目录；留空用 models_onnx/cam{N}。对比实验时指定独立目录，避免覆盖")
    args = ap.parse_args()

    base_model = args.base or BASE_MODEL
    if not os.path.exists(base_model):
        raise SystemExit("骨干模板不存在: " + base_model)

    cam_dir = os.path.join(OUT_ROOT, "cam%d" % args.cam)
    out_dir = args.out_dir or cam_dir
    meta_path = os.path.join(cam_dir, "meta_v2.json")          # ROI 始终从现场目录读
    ok_dir = args.ok_dir or OK_DIRS.get(args.cam, "")
    ng_dir = args.ng_dir or NG_DIRS.get(args.cam, "")

    if args.roi:
        roi = [int(v) for v in args.roi.split(",")]
    elif os.path.exists(meta_path):
        roi = [int(v) for v in json.load(open(meta_path, encoding="utf-8"))["roi"]]
    else:
        raise SystemExit("没有 --roi，且 %s 不存在" % meta_path)

    x1, y1, x2, y2 = roi
    files = list_images(ok_dir)
    if not files:
        raise SystemExit("没有 OK 图: " + ok_dir)

    print("=" * 74)
    print("  重建 cam%d 记忆库" % args.cam)
    print("=" * 74)
    print("  骨干模板: %.1f MB  %s" % (os.path.getsize(base_model) / 1048576, base_model))
    print("  OK 图  : %d 张  %s" % (len(files), ok_dir))
    print("  ROI    : %s  (%d x %d)" % (roi, x2 - x1, y2 - y1))
    print("  coreset: %d" % args.target)

    os.makedirs(out_dir, exist_ok=True)
    print("  输出目录: %s" % out_dir)
    front_path = os.path.join(out_dir, "front.onnx")

    t0 = time.time()
    print("\n  [1/6] 生成 front.onnx ...")
    build_front(base_model, front_path)
    print("        %.1f MB  用时 %.1fs"
          % (os.path.getsize(front_path) / 1048576, time.time() - t0))

    # ---- 提特征 ----
    print("  [2/6] 提取特征 ...")
    sess = ort.InferenceSession(front_path, providers=["CPUExecutionProvider"])
    embs = []
    t1 = time.time()
    for i, name in enumerate(files):
        x = crop_roi(os.path.join(ok_dir, name), (x1, y1, x2, y2))[None, None]
        embs.append(sess.run(None, {"gray": x})[0])
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print("        %d/%d" % (i + 1, len(files)), flush=True)
    embs = np.concatenate(embs, axis=0)
    del sess
    print("        %s  用时 %.1fs" % (embs.shape, time.time() - t1))

    n = len(files)
    ppp = PATCH_GRID * PATCH_GRID

    def take(idx):
        rows = (np.asarray(idx)[:, None] * ppp + np.arange(ppp)[None, :]).ravel()
        return embs[rows]

    # ---- 留出标定（漏检的关键：建库图自评偏乐观） ----
    print("  [3/6] 留出标定 ...")
    hold = int(min(max(10, round(n * 0.2)), 50, n - 10)) if n >= 20 else 0
    hold_mean = hold_std = hold_max = hold_thr = 0.0
    if hold >= 10:
        perm = np.random.permutation(n)
        hold_idx = np.sort(perm[:hold])
        tr_idx = np.sort(perm[hold:])
        tgt_h = max(300, int(args.target * len(tr_idx) / float(n)))
        h0 = time.time()
        bank_h = greedy_coreset(take(tr_idx), tgt_h)
        hs = knn_scores(take(hold_idx).reshape(hold, ppp, -1), bank_h)
        del bank_h
        hold_mean, hold_std = float(hs.mean()), float(hs.std())
        hold_max = float(hs.max())
        hold_thr = round(max(hold_mean + 6 * hold_std, hold_max * 1.02), 2)
        print("        建库 %d / 未入库 %d   均值 %.4f  std %.4f  max %.4f  用时 %.0fs"
              % (len(tr_idx), hold, hold_mean, hold_std, hold_max, time.time() - h0))
        print("        建议阈值 = max(mean+6σ, max×1.02) = %.2f" % hold_thr)
    else:
        print("        样本 <20，跳过留出标定（阈值会偏乐观）")

    # ---- 全量 coreset ----
    print("  [4/6] 全量 coreset %d ..." % args.target)
    c0 = time.time()
    bank = greedy_coreset(embs, args.target)
    print("        记忆库 %s  用时 %.0fs" % (bank.shape, time.time() - c0))

    # ---- 拼装 ----
    print("  [5/6] 拼装 patchcore_roi.onnx ...")
    model_path = os.path.join(out_dir, "patchcore_roi.onnx")
    build_final(front_path, bank, model_path)
    np.save(os.path.join(out_dir, "memory_bank_v2.npy"), bank)
    print("        %.1f MB" % (os.path.getsize(model_path) / 1048576))

    # ---- 自评 + NG 验证 ----
    print("  [6/6] 自评与 NG 验证 ...")
    self_scores = knn_scores(embs.reshape(n, ppp, -1), bank)
    del embs
    s_mean, s_std = float(self_scores.mean()), float(self_scores.std())
    s_max = float(self_scores.max())
    print("        建库图自评: mean %.4f std %.4f max %.4f" % (s_mean, s_std, s_max))

    thr = hold_thr if hold >= 10 else round(max(s_mean + 6 * s_std, s_max * 1.02), 2)
    print("        最终阈值 = %.2f" % thr)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    # 导出模型与 numpy 算分一致性
    diffs = []
    for i in range(min(5, n)):
        x = crop_roi(os.path.join(ok_dir, files[i]), (x1, y1, x2, y2))[None, None]
        diffs.append(abs(float(sess.run(None, {"gray": x})[0].ravel()[0]) - float(self_scores[i])))
    print("        ONNX 与 numpy 算分最大差 %.2e" % max(diffs))

    ng_scores, ng_hit = {}, 0
    if ng_dir and os.path.isdir(ng_dir):
        ng_files = list_images(ng_dir)
        for name in ng_files:
            x = crop_roi(os.path.join(ng_dir, name), (x1, y1, x2, y2))[None, None]
            ng_scores[name] = round(float(sess.run(None, {"gray": x})[0].ravel()[0]), 4)
        ng_hit = sum(1 for v in ng_scores.values() if v >= thr)
        print("        NG 检出 %d/%d" % (ng_hit, len(ng_files)))
        miss = {k: v for k, v in ng_scores.items() if v < thr}
        if miss:
            print("        未检出: %s" % json.dumps(miss, ensure_ascii=False))
    del sess

    meta = dict(site="cam%d" % args.cam, model="patchcore_roi.onnx",
                roi=roi, n_ok=n, threshold=thr, n_bank=int(bank.shape[0]),
                holdout=dict(n=hold, mean=round(hold_mean, 6),
                             std=round(hold_std, 6), max=round(hold_max, 6)),
                self_eval=dict(mean=round(s_mean, 6), std=round(s_std, 6), max=round(s_max, 6)),
                ng_hit=ng_hit, ng_total=len(ng_scores), ng_scores=ng_scores,
                backbone=backbone_name_of(base_model), input_size=224, resize=256)
    with open(os.path.join(out_dir, "meta_v2.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(site="cam%d" % args.cam, roi=roi, input_size=224, resize=256,
                       patch_shape=[PATCH_GRID, PATCH_GRID], embed_dim=int(bank.shape[1]),
                       n_bank=int(bank.shape[0]), n_neighbors=1, gaussian_sigma=4.0,
                       mask_size=[224, 224], score_reduce="max", distance="squared_l2"),
                  f, ensure_ascii=False, indent=2)

    print("\n完成，总用时 %.0fs" % (time.time() - t0))
    print("下一步: python _archive/verify_scores.py --cam %d  （注意 .bak 基准已变）" % args.cam)


if __name__ == "__main__":
    main()
