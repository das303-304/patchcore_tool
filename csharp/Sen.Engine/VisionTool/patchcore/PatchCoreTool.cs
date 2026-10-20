using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using HalconDotNet;
using Sen.Communication;
using Sen.Engine.Base;
using Sen.Engine.ImageControl;

namespace Sen.Engine.VisionTool
{
    /// <summary>
    /// PatchCore 无监督缺陷检测工具。
    ///
    /// **图像侧的用法与 Yolov5Tool 完全一致**：
    ///   - 输入 = 框架图像源 `toolGroup.GetImage(ImageInSource)` → HImage；
    ///   - 取像素 / 裁剪用 Halcon 算子（CountChannels / Rgb1ToGray / CropPart / GetImagePointer1）；
    ///   - 结果显示 = HWndCtrl.AddIconicVar(HImage / HRegion) + AddText；
    ///   - 存图 = HOperatorSet.WriteImage；读图 = HOperatorSet.ReadImage。
    ///
    /// 与 Yolov5Tool 的语义差异（必须知道）：
    ///   1. 分数方向相反：YOLO 的 Score 越大越好；PatchCore 的异常分越大越差。
    ///      写 PLC 时默认转成置信度（越大越好），保持 PLC 侧比较方向不变。
    ///   2. 缺陷区域不是检测框，而是热图超阈值后的连通域。
    ///   3. ROI 由本工具裁剪后送模型（模型内不再固化 ROI）。
    ///
    /// 判定归属与 YOLO 一致：阈值/面积/位置筛选在 C#，PLC 靠"有没有收到数据"判 NG。
    /// </summary>
    [Serializable]
    public class PatchCoreTool : VisionToolBase, IGroupShowObj
    {
        #region 供反射调用（ToolFactory 靠这些发现工具）

        public new static string GroupInfo { get; private set; } = "PatchCore";
        public new static string ToolType { get; private set; } = "PatchCore";
        public new static string Explains { get; private set; } =
            "无监督缺陷检测" + "\r\n版本:20261019" + "\r\n说明:仅用 OK 图建库，检测未见过的不良";
        public new static string ScriptKeys => GetScriptKey();

        #endregion

        public string GroupShowName => ToolName + ".PatchCore";

        [field: NonSerialized]
        private PatchCoreEngine? engine;
        [field: NonSerialized]
        private string engineSignature = "";

        /// <summary>当前参与运算的图像（来自框架图像源，或手动打开的文件）。</summary>
        [field: NonSerialized]
        private HImage? imageIn;

        [field: NonSerialized]
        private PatchCoreResult? lastResult;

        [field: NonSerialized]
        private List<PatchCoreBlob> keptBlobs = new List<PatchCoreBlob>();

        [field: NonSerialized]
        private object resultLocker = new object();

        /// <summary>
        /// 结果锁。**内部一律走这个属性，不要直接 lock(resultLocker)**。
        ///
        /// 原因：工程文件用 BinaryFormatter 存取，[NonSerialized] 字段在反序列化后
        /// 是 null，而字段初始化器只在构造函数里执行、反序列化不跑构造函数。
        /// 于是 lock(resultLocker) 会抛 ArgumentNullException（Monitor.ReliableEnter
        /// 的参数为 null）—— 打开已保存工程、双击工具即崩。
        /// 与 Yolov5Tool.SerialResultLocker / VsToolGroup.Locker 同样的自愈写法。
        /// </summary>
        private object ResultLocker
        {
            get
            {
                if (resultLocker == null)
                {
                    resultLocker = new object();
                }
                return resultLocker;
            }
        }

        /// <summary>
        /// keptBlobs 的安全访问器。同上：反序列化后是 null，
        /// 直接用会 NRE / 传给 List 构造函数会 ArgumentNullException。
        /// 只在 lock (ResultLocker) 内使用。
        /// </summary>
        private List<PatchCoreBlob> KeptBlobList
        {
            get
            {
                if (keptBlobs == null)
                {
                    keptBlobs = new List<PatchCoreBlob>();
                }
                return keptBlobs;
            }
        }

        private PatchCoreModel model = new PatchCoreModel();

        public PatchCoreModel PatchCoreDataModel
        {
            get { return model ??= new PatchCoreModel(); }
            set { model = value; }
        }

        /// <summary>输入图像源的工具名（与 YOLO 同名同义）。</summary>
        public string ImageInSource { get; set; } = "";

        /// <summary>最近一次的原始异常分（供界面显示）。</summary>
        [field: NonSerialized]
        public double LastScore { get; private set; }

        /// <summary>最近一次筛选后的缺陷数。</summary>
        public int LastDefectCount
        {
            get { lock (ResultLocker) { return KeptBlobList.Count; } }
        }

        /// <summary>实际使用的 provider（供界面/日志显示）。</summary>
        public string LastProvider => engine?.ProviderInfo ?? "未加载";

        /// <summary>最近一次结果。</summary>
        public PatchCoreResult? LastResult
        {
            get { lock (ResultLocker) { return lastResult; } }
        }

        /// <summary>最近一次筛选后的缺陷区域（供界面列表显示）。</summary>
        public List<PatchCoreBlob> KeptBlobs
        {
            get { lock (ResultLocker) { return new List<PatchCoreBlob>(KeptBlobList); } }
        }

        /// <summary>当前图像的来源说明（上游图像源工具名）。</summary>
        public string CurrentImagePath
        {
            get { return ImageInSource ?? ""; }
        }

        #region 生命周期

        public PatchCoreTool() : base() { }

        public PatchCoreTool(VsToolGroup toolGroup) : this()
        {
            ToolName = toolGroup.GenDefaultName(ToolType);
        }

        /// <summary>
        /// 覆盖基类 Run()：基类用 HOperatorSet.CountSeconds 计时，
        /// 这里改用 Stopwatch（少调一次算子），事件触发顺序与基类保持一致。
        ///
        /// 注意：基类的 ToolState 是 private set，子类无法赋值，
        /// 所以只发事件通知工具树刷新图标颜色。
        /// 这不影响 OK/NG 判定 —— VsToolGroup.GetIsOk() 读的是 tool.IsOk。
        /// </summary>
        public override OperateResult Run(VsToolGroup toolGroup)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            OperateResult result = RunOnce(toolGroup);
            sw.Stop();

            RunTime = sw.Elapsed.TotalMilliseconds;

            try
            {
                toolGroup.TriggerToolStateChangeEvent(
                    IsOk ? Sen.Engine.Base.ToolState.Ok : Sen.Engine.Base.ToolState.Ng,
                    ToolName);
            }
            catch { /* 工具树未订阅时忽略 */ }

            TriggerToolRunDoneEvent(result);
            TriggerUpdateLogEvent(result.Message);
            TriggerUpdateWndEvent("采集图像");
            return result;
        }

        public override OperateResult RunOnce(VsToolGroup toolGroup)
        {
            CollectHobj();
            IsOk = false;

            if (!string.IsNullOrEmpty(ImageInSource))
            {
                SetImageSource(toolGroup);
                if (imageIn == null || !imageIn.IsInitialized())
                    return OperateResult.CreateFailedResult<string>(
                        new OperateResult($"输入图像为空（图像源「{ImageInSource}」未产出图像）！！！"));
            }
            else
            {
                return OperateResult.CreateFailedResult<string>(
                    new OperateResult("未选择输入图像，无法运行工具！！！"));
            }

            IsOk = Calculate();

            return IsOk
                ? OperateResult.CreateSuccessResult()
                : OperateResult.CreateFailedResult<string>(new OperateResult("算法异常！！！"));
        }

        /// <summary>从框架图像源取图（与 YOLO 的用法一致）。</summary>
        public void SetImageSource(VsToolGroup toolGroup)
        {
            if (toolGroup == null) return;
            imageIn = toolGroup.GetImage(ImageInSource);
        }

        /// <summary>取当前图像（供界面/外部使用）。</summary>
        public HImage? CurrentImage => imageIn;

        /// <summary>
        /// 实际生效的模型路径。
        ///
        /// `PatchCoreDataModel.ModelPath` 为空时按**相机通道**自动生成
        /// `PatchCore\cam{N}\patchcore_roi.onnx` —— 新建工程时不用手填，
        /// 也不同担心多个相机工具指向同一个模型文件。
        /// </summary>
        public string ResolveModelPath()
        {
            var m = PatchCoreDataModel;
            string p = (m.ModelPath ?? "").Trim();
            if (p.Length == 0)
                p = $@"PatchCore\cam{m.CamChn}\patchcore_roi.onnx";
            return PatchCoreEngine.ResolvePath(p);
        }

        /// <summary>模型是否已就位（供界面显示状态）。</summary>
        public bool ModelExists => File.Exists(ResolveModelPath());

        /// <summary>建库工具产出的 meta_v2.json 内容。</summary>
        public sealed class BuildMeta
        {
            public double Threshold { get; set; }
            public double HoldMean { get; set; }
            public double HoldMax { get; set; }
            public int HoldN { get; set; }
            public int NBank { get; set; }
            /// <summary>建库时用的 ROI（原图坐标系）。null = 文件里没写。</summary>
            public int[] Roi { get; set; }
            /// <summary>meta 文件版本（最后写入时间 Ticks），用于判断是否已同步过。</summary>
            public string Stamp { get; set; } = "";
        }

        /// <summary>
        /// 读模型目录下的 meta_v2.json。读不到或没有 threshold 时返回 null。
        ///
        /// 为什么参数必须以模型为准：ROI 决定裁哪块料面、阈值决定判 NG 的门槛，
        /// 两者都是建库时定死的。C# 侧若沿用旧工程的 ROI，裁出来的图和记忆库
        /// 不是一个区域，分数会整体漂移（实测偏 12.5% 就能让正常料成片误报）。
        /// </summary>
        public BuildMeta ReadBuildMeta()
        {
            try
            {
                string dir = Path.GetDirectoryName(ResolveModelPath());
                if (string.IsNullOrEmpty(dir)) return null;
                var fi = new FileInfo(Path.Combine(dir, "meta_v2.json"));
                if (!fi.Exists) return null;

                using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(fi.FullName));
                var root = doc.RootElement;
                var meta = new BuildMeta();
                if (root.TryGetProperty("threshold", out var t)) meta.Threshold = t.GetDouble();
                if (root.TryGetProperty("n_bank", out var nb)) meta.NBank = nb.GetInt32();
                if (root.TryGetProperty("holdout", out var h))
                {
                    if (h.TryGetProperty("mean", out var hm)) meta.HoldMean = hm.GetDouble();
                    if (h.TryGetProperty("max", out var hx)) meta.HoldMax = hx.GetDouble();
                    if (h.TryGetProperty("n", out var hn)) meta.HoldN = hn.GetInt32();
                }
                if (root.TryGetProperty("roi", out var roiEl)
                    && roiEl.ValueKind == System.Text.Json.JsonValueKind.Array
                    && roiEl.GetArrayLength() == 4)
                {
                    var arr = new int[4];
                    for (int i = 0; i < 4; i++) arr[i] = roiEl[i].GetInt32();
                    if (arr[2] > arr[0] && arr[3] > arr[1]) meta.Roi = arr;
                }
                meta.Stamp = fi.LastWriteTimeUtc.Ticks.ToString();
                return meta.Threshold > 0 ? meta : null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>兼容旧调用：只取阈值。</summary>
        public bool TryReadBuildMeta(out double threshold, out double holdMax, out int holdN)
        {
            var meta = ReadBuildMeta();
            if (meta == null)
            {
                threshold = 0; holdMax = 0; holdN = 0;
                return false;
            }
            threshold = meta.Threshold;
            holdMax = meta.HoldMax;
            holdN = meta.HoldN;
            return true;
        }

        /// <summary>丢弃当前引擎，下次 Calculate 时按新路径重建（导入/替换模型后调用）。</summary>
        public void ResetEngine()
        {
            engine?.Dispose();
            engine = null;
            engineSignature = "";
        }

        /// <summary>
        /// 惰性创建/重建引擎。模型路径、ROI 或 GPU 开关变化时会重建。
        /// </summary>
        private PatchCoreEngine EnsureEngine()
        {
            var m = PatchCoreDataModel;
            string path = ResolveModelPath();
            string sig = path + "|" + string.Join(",", m.RoiCopy()) + "|" + m.PreferGpu;

            if (engine != null && engineSignature == sig) return engine;

            engine?.Dispose();
            engine = new PatchCoreEngine(path, m.RoiCopy(), m.PreferGpu);
            engineSignature = sig;

            TriggerUpdateLogEvent("PatchCore 模型加载：" + engine.ProviderInfo);

            if (!engine.Ready)
            {
                // 初建工程最常见的情况：还没给这个相机建库
                TriggerUpdateLogEvent($"  期望路径：{path}");
                TriggerUpdateLogEvent("  → 每个相机必须使用自己建库的模型，不能共用（记忆库绑定该工位的成像条件）");
                TriggerUpdateLogEvent("  → 建库后把 patchcore_roi.onnx 放到上面这个路径，或在界面里改成实际路径");
            }

            return engine;
        }

        protected override bool Calculate()
        {
            if (imageIn == null || !imageIn.IsInitialized()) return false;

            var m = PatchCoreDataModel;
            PatchCoreEngine eng;
            try
            {
                eng = EnsureEngine();
            }
            catch (Exception ex)
            {
                TriggerUpdateLogEvent("模型加载失败：" + ex.Message);
                return false;
            }

            if (!eng.Ready)
            {
                TriggerUpdateLogEvent("模型不可用：" + eng.ProviderInfo);
                return false;
            }

            PatchCoreResult r;
            try
            {
                r = eng.Run(imageIn, m.Threshold, m.BlobThresholdRatio,
                            m.MinArea, m.MaxArea, m.MaxBlobs);
            }
            catch (Exception ex)
            {
                TriggerUpdateLogEvent("推理失败：" + ex.Message);
                return false;
            }

            var kept = FilterByPosition(r.Blobs, m);
            kept = kept.OrderByDescending(b => b.Area).ToList();

            lock (ResultLocker)
            {
                lastResult = r;
                keptBlobs = kept;
            }
            LastScore = r.Score;

            TriggerUpdateLogEvent(
                $"异常分 {r.Score:F3}（阈值 {m.Threshold:F2}）  " +
                $"区域 {kept.Count} 个  {r.ElapsedMs} ms  {CurrentImagePath}");

            if (m.SaveOkImage && kept.Count == 0)
                Task.Run(() => SaveImage(true, m.CamChn, m.PicSaveName));

            return true;   // 与 YOLO 一致：本工具只报"算法是否成功"，NG 由 PLC 依据数据判定
        }

        /// <summary>
        /// 位置范围过滤，语义与 YOLO 一致：
        /// 要求 X+W ≥ XLeftLimit、X ≤ XRightLimit、Y ≥ YTopLimit、Y+H ≤ YBottomLimit。
        /// 默认值全 0 时视为"不启用"（否则会把所有区域滤掉）。
        /// </summary>
        private static List<PatchCoreBlob> FilterByPosition(List<PatchCoreBlob> src, PatchCoreModel m)
        {
            bool enabled = m.XRightLimit > 0 || m.YBottomLimit > 0;
            if (!enabled) return new List<PatchCoreBlob>(src);

            var kept = new List<PatchCoreBlob>();
            foreach (var b in src)
            {
                if (b.X + b.W < m.XLeftLimit) continue;
                if (b.X > m.XRightLimit) continue;
                if (b.Y < m.YTopLimit) continue;
                if (b.Y + b.H > m.YBottomLimit) continue;
                kept.Add(b);
            }
            return kept;
        }

        #endregion

        #region 显示（与 YOLO 同一套 Halcon 画法）

        public override void AddShowHObjects(HWndCtrl mView)
        {
            if (imageIn != null && imageIn.IsInitialized())
                mView.AddIconicVar(imageIn);

            mView.ChangeGraphicSettings(GraphicsContext.GC_DRAWMODE, "margin");
            mView.ChangeGraphicSettings(GraphicsContext.GC_LINEWIDTH, 2);

            // ROI 框（绿）
            DrawRoi(mView, "green");

            lock (ResultLocker)
            {
                foreach (var b in KeptBlobList)
                {
                    mView.ChangeGraphicSettings(GraphicsContext.GC_COLOR, "red");
                    var region = new HRegion((double)b.Y, (double)b.X,
                                             (double)(b.Y + b.H), (double)(b.X + b.W));
                    try
                    {
                        mView.AddIconicVar(region);
                    }
                    finally
                    {
                        region.Dispose();
                    }
                    mView.AddText($"{b.Peak:F2},{b.Area}", Math.Max(0, b.Y - 40), b.X, 20, "yellow");
                }
            }

            mView.Repaint();
        }

        public void AddGroupShowOjbs(HWndCtrl mView, string ToolName)
        {
            AddShowHObjects(mView);

            var m = PatchCoreDataModel;
            lock (ResultLocker)
            {
                if (KeptBlobList.Count > 0) WriteToPlc(m, KeptBlobList);
            }

            if (m.SaveNgImage && LastDefectCount != 0)
                Task.Run(() => SaveImage(false, m.CamChn, m.PicSaveName));

            mView.Repaint(true);
        }

        private void DrawRoi(HWndCtrl mView, string color)
        {
            var m = PatchCoreDataModel;
            if (m.Roi == null || m.Roi.Length != 4) return;
            if (m.Roi[2] <= m.Roi[0] || m.Roi[3] <= m.Roi[1]) return;

            mView.ChangeGraphicSettings(GraphicsContext.GC_COLOR, color);
            mView.ChangeGraphicSettings(GraphicsContext.GC_DRAWMODE, "margin");
            var roi = new HRegion((double)m.Roi[1], (double)m.Roi[0],
                                  (double)m.Roi[3], (double)m.Roi[2]);
            try
            {
                mView.AddIconicVar(roi);
            }
            finally
            {
                roi.Dispose();
            }
        }

        #endregion

        #region PLC 输出（沿用 YOLO 的寄存器协议）

        private void WriteToPlc(PatchCoreModel m, List<PatchCoreBlob> blobs)
        {
            double coef = m.PixelLocCoef;
            var payload = new string[blobs.Count * 5 + 1];
            payload[0] = (blobs.Count * 5).ToString();

            int idx = 1;
            foreach (var b in blobs)
            {
                payload[idx++] = (b.X * coef * 10).ToString("F0");
                payload[idx++] = ((m.CutToCamDist + b.Y * coef) * 10).ToString("F0");
                payload[idx++] = (b.W * coef * 10).ToString("F0");
                payload[idx++] = (b.H * coef * 10).ToString("F0");

                double value = m.UseConfidence
                    ? PatchCoreEngine.ToConfidence(b.Peak, m.Threshold, m.ConfidenceByMean, m.NormalMean)
                    : b.Peak;
                payload[idx++] = (value * 1000).ToString("F0");
            }

            int baseAddr = m.CamChn switch
            {
                1 => 2000,
                2 => 3000,
                3 => 4000,
                4 => 5000,
                _ => 2000
            };

            var data = Project.Instance?.ProjectManage?.Data;
            if (data?.CommDrivers == null) return;

            foreach (var item in data.CommDrivers)
            {
                if (item.ToolName != m.CommToolName) continue;
                if (item is not Sen.Engine.Communication.ThingetPLC plc) continue;

                plc.WriteMultD(baseAddr, payload);
                System.Threading.Thread.Sleep(20);

                // 可选：Y 方向极值（最高/最低缺陷），单独写到 基址+offset
                if (m.OutputYExtent)
                {
                    var yPayload = new string[4];
                    yPayload[0] = "3";
                    yPayload[1] = ((m.CutToCamDist + blobs.Min(b => b.Y) * coef) * 10).ToString("F0");
                    yPayload[2] = ((m.CutToCamDist + blobs.Max(b => b.Y + b.H) * coef) * 10).ToString("F0");
                    yPayload[3] = blobs.Count.ToString();
                    plc.WriteMultD(baseAddr + m.YExtentAddressOffset, yPayload);
                    System.Threading.Thread.Sleep(20);
                }
                return;
            }
        }

        #endregion

        #region 存图（与 YOLO 相同规则，Halcon 落盘）

        private void SaveImage(bool resultOk, int camChn, string picName)
        {
            try
            {
                if (imageIn == null || !imageIn.IsInitialized()) return;

                string root = resultOk
                    ? ImageSaveGlobalConfig.OkImageRoot
                    : ImageSaveGlobalConfig.NgImageRoot;

                string dir = Path.Combine(root, DateTime.Now.ToString("yyyy-MM"));
                dir = Path.Combine(dir, DateTime.Now.ToString("yyyy-MM-dd"));
                dir = Path.Combine(dir, picName + "相机" + camChn);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

                int count = Directory.GetFiles(dir).Length + 1;
                string file = Path.Combine(dir,
                    count + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));

                HOperatorSet.WriteImage(imageIn, "jpg", 0, file);
                ImageSaveGlobalConfig.CleanupExpiredImages();
            }
            catch (Exception ex)
            {
                TriggerUpdateLogEvent("存图失败：" + ex.Message);
            }
        }

        #endregion

        #region 框架接口

        public override System.Windows.Forms.UserControl GetToolUI(VsToolGroup toolGroup)
        {
            toolSetting ??= new UcPatchCoreTool(this, toolGroup);
            return toolSetting;
        }

        public override void CollectHobj()
        {
            // 本工具不长期持有 HRegion（画完即释放），无需回收。
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                engine?.Dispose();
                engine = null;
            }
            base.Dispose(disposing);
        }

        public static string GetScriptKey()
        {
            return "PatchCoreTool ";
        }

        #endregion
    }
}
