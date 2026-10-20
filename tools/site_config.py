"""多机台部署配置管理 —— 解决「每台机环境光/打光/物距/ROI 都可能不同」。

核心思路：
  1. 把「成像条件」显式参数化，每股差异都写成配置文件里的一个字段
  2. 每台机器一套独立配置 + 独立记忆库，互不干扰
  3. 提供一致性检查：上线前就告诉你这台机器的成像条件偏离建库机多少

配置文件结构（sites/<站点名>.json）:
{
  "site": "line1_station3",
  "camera": {"model": "MV-CU120-10GM", "serial": "DA1234567",
             "resolution": [4024, 3036], "exposure_us": 3000, "gain": 8.0},
  "lighting": {"type": "bar_led_white", "angle_deg": 45, "distance_mm": 120,
               "intensity_pct": 70, "shielded": true},
  "optics": {"working_distance_mm": 350, "lens_mm": 25, "mm_per_pixel": 0.0421},
  "roi": {"mode": "fixed_mm", "x_mm": 12.0, "y_mm": 8.0,
          "w_mm": 40.0, "h_mm": 30.0},
  "model": {"dir": "my_models/line1_station3", "threshold": 1.10},
  "normalization": {"white_balance": true, "gray_stretch": true,
                    "target_mean": 128, "target_std": 40}
}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SITES_DIR = os.path.join(HERE, "sites")

# 建库机的「黄金成像条件」—— 其他机器向它对齐
REFERENCE = {
    "lighting_angle_deg": 45,
    "working_distance_mm": 350,
    "mm_per_pixel": 0.0421,
    "exposure_us": 3000,
}

# 各项的容忍带宽（超出则判定为「必须单独建库」）
TOLERANCE = {
    "lighting_angle_deg": 5.0,      # 打光角度 ±5°
    "working_distance_mm": 15.0,    # 物距 ±15mm
    "mm_per_pixel": 0.002,          # 尺度 ±4.7%
    "exposure_us": 500.0,           # 曝光 ±500us
}


def load_site(name):
    p = os.path.join(SITES_DIR, f"{name}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"站点配置不存在: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_sites():
    if not os.path.isdir(SITES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SITES_DIR) if f.endswith(".json"))


def check_against_reference(cfg, reference=None):
    """检查某站点的成像条件相对建库机的偏离程度。

    返回 (可复用共享库?, 逐项偏离报告)
    """
    ref = {**REFERENCE, **(reference or {})}
    lt, op, cam = cfg.get("lighting", {}), cfg.get("optics", {}), cfg.get("camera", {})

    actual = {
        "lighting_angle_deg": lt.get("angle_deg"),
        "working_distance_mm": op.get("working_distance_mm"),
        "mm_per_pixel": op.get("mm_per_pixel"),
        "exposure_us": cam.get("exposure_us"),
    }

    report = []
    worst_ratio = 0.0
    for k, act in actual.items():
        if act is None:
            report.append((k, None, ref[k], None, "未填写"))
            continue
        dev = abs(act - ref[k])
        tol = TOLERANCE[k]
        ratio = dev / tol if tol else 0
        worst_ratio = max(worst_ratio, ratio)
        if ratio <= 1.0:
            verdict = "OK"
        elif ratio <= 2.0:
            verdict = "临界"
        else:
            verdict = "超标"
        report.append((k, act, ref[k], dev, verdict))

    return worst_ratio <= 1.0, report, worst_ratio


def make_template(name):
    """生成一个站点配置模板。"""
    return {
        "site": name,
        "camera": {"model": "", "serial": "", "resolution": [0, 0],
                   "exposure_us": 3000, "gain": 8.0},
        "lighting": {"type": "", "angle_deg": 45, "distance_mm": 120,
                     "intensity_pct": 70, "shielded": True},
        "optics": {"working_distance_mm": 350, "lens_mm": 25,
                   "mm_per_pixel": 0.0421},
        "roi": {"mode": "fixed_mm", "x_mm": 0, "y_mm": 0, "w_mm": 40, "h_mm": 30},
        "model": {"dir": f"my_models/{name}", "threshold": None},
        "normalization": {"white_balance": True, "gray_stretch": True,
                          "target_mean": 128, "target_std": 40},
    }


def save_site(cfg):
    os.makedirs(SITES_DIR, exist_ok=True)
    p = os.path.join(SITES_DIR, f"{cfg['site']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return p


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        # 生成一个示例：把你的建库机配置填进去
        for nm in ["line1_station3", "line2_station1", "line2_station2"]:
            print(f"生成模板: {save_site(make_template(nm))}")
        sys.exit(0)

    print("=" * 78)
    print("多机台部署一致性检查")
    print("=" * 78)
    print(f"\n建库机黄金条件（其他机器需向此对齐）:")
    for k, v in REFERENCE.items():
        print(f"  {k:24s} {v}   （容忍 ±{TOLERANCE[k]}）")

    sites = list_sites()
    if not sites:
        print(f"\n未发现站点配置。先跑:  python site_config.py --init")
        print(f"配置目录: {SITES_DIR}")
        sys.exit(0)

    print(f"\n发现 {len(sites)} 个站点配置: {', '.join(sites)}\n")
    for nm in sites:
        cfg = load_site(nm)
        ok, rep, worst = check_against_reference(cfg)
        status = "可复用共享库" if ok else f"需单独建库（最大偏离 {worst:.1f}x 容差）"
        print(f"--- {nm} ---  {status}")
        for k, act, ref, dev, verdict in rep:
            if act is None:
                print(f"    {k:24s} 未填写")
            else:
                print(f"    {k:24s} 实际 {act:<10} 基准 {ref:<10} "
                      f"偏离 {dev:<8.4g} [{verdict}]")
        print()
