"""自定义数据集适配器 - 用你自己的皮料建 PatchCore 记忆库（无需标注）。

MVTecDataset 硬编码了 15 个 MVTec 类别名，且假定目录结构为
    <root>/<class>/train/<type>/*.png
本模块提供等价但更宽松的数据集，直接指向你自己的目录：

布局 A（推荐，只有正常样本，无需任何标注）:
    leather/
      good/            # 正常皮料 —— 只需这个
        a001.png
        a002.png

布局 B（有缺陷样本，想量化检出力，需要像素级标注）:
    leather/
      good/            # 正常
      scratch/         # 缺陷 1
      stain/           # 缺陷 2
      ground_truth/    # 像素级掩码，按缺陷类型分文件夹
        scratch/
          a001_mask.png
        stain/
          ...

关键事实（已核对 patchcore.py:96 / mvtec.py:146）:
    建记忆库只读正常样本，ground_truth 仅在 TEST split 且 anomaly != good 时读取。
    所以只做「部署上线」的话，连缺陷样本都不需要 —— 这正是无监督的价值。
"""
import os

import PIL
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class CustomDataset(torch.utils.data.Dataset):
    """通用单类别缺陷检测数据集，兼容 PatchCore 的 fit/predict 接口。"""

    def __init__(
        self,
        source,
        classname="leather",
        resize=256,
        imagesize=224,
        mode="train",          # "train" 只用正常样本；"test" 加载全部+掩码
        good_dirname="good",
        mask_dirname="ground_truth",
        augment=False,
        roi=None,              # (x0, y0, x1, y1) 原图像素坐标，裁掉背景/黑边
    ):
        super().__init__()
        self.source = source
        self.classname = classname
        self.mode = mode
        self.good_dirname = good_dirname
        self.mask_dirname = mask_dirname
        self.imagesize = (3, imagesize, imagesize)
        self.roi = tuple(roi) if roi else None

        self.root = os.path.join(source, classname)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"类别目录不存在: {self.root}")

        self.data_to_iterate = self._scan()

        img_tf = []
        if self.roi:
            # ROI 必须在 Resize 之前、原图坐标系下裁，才能真正去掉黑边
            img_tf.append(transforms.Lambda(lambda im: im.crop(self.roi)))
        img_tf.append(transforms.Resize(resize))
        if augment:
            # 工业现场实际变化：位置漂移、光照波动、轻微旋转
            img_tf.append(
                transforms.RandomAffine(
                    degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
                )
            )
        img_tf += [
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(img_tf)

        mask_tf = []
        if self.roi:
            mask_tf.append(transforms.Lambda(lambda im: im.crop(self.roi)))
        mask_tf += [transforms.Resize(resize), transforms.CenterCrop(imagesize),
                    transforms.ToTensor()]
        self.transform_mask = transforms.Compose(mask_tf)

    # ---------- 扫描 ----------
    @staticmethod
    def _list_images(folder):
        if not os.path.isdir(folder):
            return []
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXT
        )

    def _scan(self):
        good_dir = os.path.join(self.root, self.good_dirname)
        good_imgs = self._list_images(good_dir)
        if not good_imgs:
            raise FileNotFoundError(
                f"未在 {good_dir} 找到正常样本图片。\n"
                f"请把正常皮料图片放到该目录（支持 {'/'.join(sorted(IMG_EXT))}）"
            )

        if self.mode == "train":
            # 只返回正常样本，无掩码
            return [[self.classname, "good", p, None] for p in good_imgs]

        # test 模式：正常 + 所有缺陷类型
        items = [[self.classname, "good", p, None] for p in good_imgs]
        for entry in sorted(os.listdir(self.root)):
            if entry in (self.good_dirname, self.mask_dirname):
                continue
            d = os.path.join(self.root, entry)
            if not os.path.isdir(d):
                continue
            mask_dir = os.path.join(self.root, self.mask_dirname, entry)
            masks = self._list_images(mask_dir)
            imgs = self._list_images(d)
            for i, p in enumerate(imgs):
                # 掩码数量与图片不一致时退化为无掩码（仅计图像级指标）
                m = masks[i] if i < len(masks) else None
                items.append([self.classname, entry, p, m])
        return items

    # ---------- PyTorch 接口 ----------
    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = self.transform_img(PIL.Image.open(image_path).convert("RGB"))

        if mask_path is not None and os.path.exists(mask_path):
            mask = self.transform_mask(PIL.Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != self.good_dirname),
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }

    def stats(self):
        """返回样本统计，便于训练前确认数据量是否够。"""
        from collections import Counter

        c = Counter(x[1] for x in self.data_to_iterate)
        return dict(c)
