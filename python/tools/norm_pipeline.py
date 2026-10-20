"""跨机台图像归一化 —— 把不同机器拍到的图「拉齐」成同一套成像条件。

这是第二层防线的实现。核心是三个归一化，顺序不能变：
  1. 尺度归一 (scale)      —— 按 mm/px 重采样，让 1 个模型像素 = 固定物理尺寸
  2. 位置归一 (locate)     —— 在产品上找基准特征，按产品坐标系裁 ROI
  3. 亮度归一 (photometric) —— 灰度拉伸/白平衡，消曝光与色温差异

设计原则：**推理时必须和建库时用完全相同的归一化参数**，
所以归一化参数要随记忆库一起保存（norm_params.json）。

用法:
    from norm_pipeline import Normalizer
    n = Normalizer.load("my_models/line1_station3")
    img_norm = n(img)              # PIL.Image -> PIL.Image (已归一化)
    score, mask = det.predict_array(np.array(img_norm))
"""
import json
import os

import numpy as np
import PIL.Image


class Normalizer:
    """跨机台图像归一化器。

    参数固化在 norm_params.json，建库与推理必须共用同一份。
    """

    def __init__(self, *, mm_per_pixel=0.0421, target_mm_per_pixel=0.0421,
                 target_size=(224, 224), roi_mm=None,
                 locator=None, gray_stretch=True, white_balance=True,
                 target_mean=128.0, target_std=40.0):
        self.mm_per_pixel = mm_per_pixel
        self.target_mm_per_pixel = target_mm_per_pixel
        self.target_size = target_size
        self.roi_mm = roi_mm                  # (x, y, w, h) 产品坐标系下的物理 ROI
        self.locator = locator                # 可调用对象: img -> (cx, cy)
        self.gray_stretch = gray_stretch
        self.white_balance = white_balance
        self.target_mean = target_mean
        self.target_std = target_std

    # ---------- 1. 尺度归一 ----------
    def rescale(self, img):
        """按物理尺寸重采样，抵消物距/焦距差异。"""
        if self.target_mm_per_pixel is None or self.mm_per_pixel is None:
            return img
        f = self.mm_per_pixel / self.target_mm_per_pixel
        if abs(f - 1.0) < 1e-3:
            return img
        w, h = img.size
        nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        return img.resize((nw, nh), PIL.Image.LANCZOS)

    # ---------- 2. 位置归一 ----------
    def crop_roi(self, img):
        """按产品坐标系裁 ROI，抵消产品在画面中位置不同的差异。"""
        if not self.roi_mm:
            return img
        x_mm, y_mm, w_mm, h_mm = self.roi_mm

        # 有定位器就先找到产品基准点，把 ROI 锚在基准点上
        if self.locator is not None:
            cx, cy = self.locator(img)
        else:
            cx, cy = img.size[0] / 2, img.size[1] / 2   # 退化：画面中心

        mpp = self.mm_per_pixel or 1.0
        x0 = int(round(cx + x_mm / mpp))
        y0 = int(round(cy + y_mm / mpp))
        w = int(round(w_mm / mpp))
        h = int(round(h_mm / mpp))
        return img.crop((x0, y0, x0 + w, y0 + h))

    # ---------- 3. 亮度归一 ----------
    def photometric(self, img):
        """消曝光/色温差异。白平衡 -> 灰度拉伸，顺序不能反。"""
        a = np.asarray(img).astype(np.float32)

        if self.white_balance:
            # 灰度世界假设：让 R/G/B 通道均值相等
            means = a.reshape(-1, 3).mean(axis=0)
            target = means.mean()
            if np.all(means > 1e-6):
                a *= (target / means).reshape(1, 1, 3)

        a = np.clip(a, 0, 255)

        if self.gray_stretch:
            # 把整体亮度和对比度拉到固定目标，消除曝光差异
            g = a.mean(axis=2)
            m, s = g.mean(), g.std()
            if s > 1e-6:
                gain = self.target_std / s
                a = (a - m) * gain + self.target_mean

        a = np.clip(a, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(a)

    # ---------- 主流程 ----------
    def __call__(self, img, return_intermediate=False):
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)
        img = img.convert("RGB")

        s = self.rescale(img)
        c = self.crop_roi(s)
        p = self.photometric(c)
        if self.target_size:
            p = p.resize(self.target_size, PIL.Image.LANCZOS)

        if return_intermediate:
            return p, dict(original=img, rescaled=s, roi=c,
                           photometric=p.resize(c.size, PIL.Image.LANCZOS))
        return p

    # ---------- 持久化 ----------
    def to_dict(self):
        return {
            "mm_per_pixel": self.mm_per_pixel,
            "target_mm_per_pixel": self.target_mm_per_pixel,
            "target_size": list(self.target_size) if self.target_size else None,
            "roi_mm": list(self.roi_mm) if self.roi_mm else None,
            "gray_stretch": self.gray_stretch,
            "white_balance": self.white_balance,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        p = os.path.join(model_dir, "norm_params.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return p

    @classmethod
    def load(cls, model_dir, locator=None):
        p = os.path.join(model_dir, "norm_params.json")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"未找到归一化参数: {p}\n"
                f"建库时请调用 normalizer.save(model_dir)，"
                f"否则推理无法复现建库时的成像条件。")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["target_size"] = tuple(d["target_size"]) if d["target_size"] else None
        d["roi_mm"] = tuple(d["roi_mm"]) if d["roi_mm"] else None
        return cls(locator=locator, **d)


def estimate_mm_per_pixel(reference_mm, pixel_size, axis="x"):
    """由已知物理尺寸的标定物推算 mm/px。

    reference_mm: 标定物的真实物理尺寸（毫米）
    pixel_size:   该标定物在图像中占的像素数
    """
    if pixel_size <= 0:
        raise ValueError("pixel_size 必须为正")
    return reference_mm / pixel_size


if __name__ == "__main__":
    print(__doc__)
    print("=" * 70)
    print("自检：模拟三台机器拍同一块料")
    print("=" * 70)

    # 造一张带纹理的假图，模拟同一块料在三种成像条件下的样子
    rng = np.random.default_rng(0)
    base = (rng.normal(150, 18, (400, 400, 3))).clip(0, 255)
    base[180:220, :, :] -= 60          # 一条横向缺陷
    base_img = PIL.Image.fromarray(base.astype(np.uint8))

    configs = {
        "机台A 基准":   dict(mm_per_pixel=0.0421),
        "机台B 物距近": dict(mm_per_pixel=0.0330),   # 尺度变了
        "机台C 曝光高": dict(mm_per_pixel=0.0421),   # 亮度变了
    }

    for name, kw in configs.items():
        img = base_img
        if "曝光" in name:
            a = np.asarray(img).astype(np.float32) * 1.35 + 20
            img = PIL.Image.fromarray(a.clip(0, 255).astype(np.uint8))

        # 关键：mm_per_pixel 描述的是「原图」，归一化目标是回到基准
        n = Normalizer(mm_per_pixel=kw["mm_per_pixel"],
                       target_mm_per_pixel=0.0421,
                       target_size=(224, 224),
                       roi_mm=(-40, -40, 80, 80))
        out, mid = n(img, return_intermediate=True)
        a = np.asarray(out).astype(np.float32)
        print(f"\n{name}")
        print(f"  原图尺寸      {img.size}  归一后 {out.size}")
        print(f"  归一后均值/标准差  {a.mean():6.2f} / {a.std():5.2f}"
              f"   （目标 {128:.0f} / {40:.0f}）")
    print("\n三台机器归一化后的统计量应高度接近 —— 这才具备可比性。")
