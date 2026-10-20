using System;

namespace Sen.Engine.VisionTool
{
    /// <summary>
    /// PatchCore 工具的参数模型。
    /// 全部是值类型/字符串，可被 BinaryFormatter 序列化进工程文件。
    /// </summary>
    [Serializable]
    public class PatchCoreModel
    {
        /// <summary>
        /// 模型文件（相对 XVision.exe 目录，或绝对路径）。
        ///
        /// **留空 = 按相机通道自动生成** `PatchCore\cam{N}\patchcore_roi.onnx`。
        /// 这样多个相机工具不需要各自手填路径，也不会互相覆盖。
        ///
        /// 注意：每个相机必须用**自己建库**得到的模型。记忆库绑定的是该工位的
        /// 成像条件（光源/物距/ROI），拿 A 机的模型去跑 B 机必然成片误报。
        /// </summary>
        public string ModelPath { get; set; } = "";

        /// <summary>
        /// 最近一次检测的图片来源说明（图像源工具名，或"文件:xxx.jpg"），仅用于界面显示。
        /// 实际取图走 Halcon：框架图像源 toolGroup.GetImage()，或 ReadImage 读本地文件。
        /// </summary>
        public string CurrentImagePath { get; set; } = "";

        /// <summary>ROI：x0, y0, x1, y1（原图像素坐标，右/下为开区间）。</summary>
        public int[] Roi { get; set; } = new[] { 540, 30, 1780, 1270 };

        /// <summary>异常分阈值，越大越差。</summary>
        public double Threshold { get; set; } = 0.59;

        /// <summary>热图二值化阈值 = Threshold × 该系数。</summary>
        public double BlobThresholdRatio { get; set; } = 1.0;

        /// <summary>缺陷区域最小面积（原图像素²）。</summary>
        public int MinArea { get; set; } = 200;

        /// <summary>缺陷区域最大面积，0 = 不限。</summary>
        public int MaxArea { get; set; }

        /// <summary>单张最多输出几条缺陷（按面积从大到小）。</summary>
        public int MaxBlobs { get; set; } = 10;

        /// <summary>写入 PLC 的是否为置信度（true）而非原始异常分（false）。</summary>
        public bool UseConfidence { get; set; } = true;

        /// <summary>
        /// 置信度公式：false = clamp(1 - score/阈值, 0, 1)；
        /// true = clamp((阈值 - score)/(阈值 - 正常均值), 0, 1)（正常料接近 1000）。
        /// </summary>
        public bool ConfidenceByMean { get; set; }

        /// <summary>正常样本平均分，仅 ConfidenceByMean = true 时使用。</summary>
        public double NormalMean { get; set; } = 0.49;

        /// <summary>
        /// 已经应用过的模型 meta_v2.json 版本（文件最后写入时间的 Ticks）。
        /// 换模型（导入/替换 onnx）时该值会变，据此自动同步 ROI / 阈值 / 正常均值；
        /// 用户之后自己调的参数不会被反复覆盖。
        /// </summary>
        public string MetaStamp { get; set; } = "";

        /// <summary>是否额外输出 Y 方向极值（最高/最低缺陷）。</summary>
        public bool OutputYExtent { get; set; }

        /// <summary>Y 极值写到 基址 + 该偏移 的 3 个寄存器。</summary>
        public int YExtentAddressOffset { get; set; } = 100;

        /// <summary>
        /// 是否优先尝试 GPU（不可用会自动回退 CPU）。
        ///
        /// 默认 true：CUDA 11.8 运行时已随程序部署（exe 同目录），GTX 1650 / RTX 3060 都覆盖；
        /// 环境不满足时 AppendExecutionProvider_CUDA 抛异常 → 自动回退 CPU，不会崩。
        /// 现场若只想用 CPU（比如排故），把界面上的「优先GPU」取消勾选即可。
        /// </summary>
        public bool PreferGpu { get; set; } = true;

        /// <summary>写 PLC 时匹配的通信工具名，与 YOLO 工具保持一致。</summary>
        public string CommToolName { get; set; } = "Serial服务器1";

        public bool SaveOkImage { get; set; }
        public bool SaveNgImage { get; set; }
        public string PicSaveName { get; set; } = "";

        /// <summary>相机通道：1→D2000, 2→D3000, 3→D4000, 4→D5000。</summary>
        public int CamChn { get; set; } = 1;

        /// <summary>切料到相机处的位置（mm），与 YOLO 工具同义。</summary>
        public double CutToCamDist { get; set; }

        /// <summary>像素转 mm 的系数。</summary>
        public double PixelLocCoef { get; set; }

        public double XLeftLimit { get; set; }
        public double XRightLimit { get; set; }
        public double YTopLimit { get; set; }
        public double YBottomLimit { get; set; }

        /// <summary>深拷贝一份 ROI，避免外部修改共享数组。</summary>
        public int[] RoiCopy()
        {
            if (Roi == null || Roi.Length != 4) return new[] { 0, 0, 0, 0 };
            return new[] { Roi[0], Roi[1], Roi[2], Roi[3] };
        }
    }
}
