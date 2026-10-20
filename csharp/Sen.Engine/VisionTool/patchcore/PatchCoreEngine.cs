using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using HalconDotNet;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Sen.Engine.VisionTool
{
    /// <summary>单个缺陷区域（原图坐标）。</summary>
    public sealed class PatchCoreBlob
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int W { get; set; }
        public int H { get; set; }
        public int Area { get; set; }
        public double Peak { get; set; }

        public int X2 => X + W;
        public int Y2 => Y + H;
    }

    /// <summary>单张图的推理结果。</summary>
    public sealed class PatchCoreResult
    {
        public double Score { get; set; }
        public float[] Map { get; set; } = Array.Empty<float>();
        public List<PatchCoreBlob> Blobs { get; set; } = new List<PatchCoreBlob>();
        public long ElapsedMs { get; set; }
        public int RoiWidth { get; set; }
        public int RoiHeight { get; set; }
        public int RoiX { get; set; }
        public int RoiY { get; set; }
    }

    /// <summary>
    /// PatchCore 推理封装（与工具类解耦，便于单独测试）。
    ///
    /// 链路：HImage → 灰度 → 按 ROI 整数裁剪 → uint8 张量 → ONNX
    ///       → score + anomaly_map(224×224) → 阈值连通域 → 原图坐标
    ///
    /// 图像侧全部走 Halcon（与 YOLO 工具一致）：
    ///   CountChannels / Rgb1ToGray / GetImageSize / CropPart / GetImagePointer1。
    /// ROI 裁剪必须是整数 CropPart，不能重采样 —— 模型内部做
    /// 「256×256 抗锯齿缩放 → 中心裁 224 → 归一化」，C# 侧一旦插值就与训练侧对不上
    /// （实测用非抗锯齿的 LINEAR 代替，正常料的分数会翻倍并被判成 NG）。
    /// </summary>
    public sealed class PatchCoreEngine : IDisposable
    {
        public const int MapSize = 224;
        private const int ResizeTo = 256;
        private const int CropOffset = 16;      // (256 - 224) / 2

        private readonly object _lock = new object();
        private InferenceSession? _session;
        private string _modelPath = "";
        private int[] _roi = Array.Empty<int>();

        /// <summary>实际生效的 provider 描述（用于日志，避免"以为在用 GPU"）。</summary>
        public string ProviderInfo { get; private set; } = "未初始化";

        public bool Ready => _session != null;

        /// <summary>
        /// 创建引擎。preferGpu = true 时先尝试 CUDA，失败自动回退 CPU。
        /// </summary>
        public PatchCoreEngine(string modelPath, int[] roi, bool preferGpu = false)
        {
            _modelPath = ResolvePath(modelPath);
            _roi = roi ?? Array.Empty<int>();

            if (!System.IO.File.Exists(_modelPath))
            {
                ProviderInfo = "模型文件不存在: " + _modelPath;
                return;
            }

            var sw = Stopwatch.StartNew();
            if (preferGpu)
            {
                try
                {
                    var soGpu = new SessionOptions
                    {
                        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                    };

                    // 不能只调 AppendExecutionProvider_CUDA(deviceId) —— 那样用的是默认 arena，
                    // ORT 会按 kNextPowerOfTwo 反复翻倍扩池，在 GTX 1650 这种 4GB 卡上容易把
                    // 显存啃光（本机 3060 12GB 看不出问题，现场 1650 会直接 OOM 或拖慢显示）。
                    //
                    // gpu_mem_limit     : 硬上限 2GB，给显卡输出/其他程序留出余量
                    // arena_extend_strategy=kSameAsRequested : 按需增长，不做倍数扩张
                    // cudnn_conv_algo_search=HEURISTIC       : 不做穷举搜索，会话创建快（EXHAUSTIVE 首帧要数秒）
                    // do_copy_in_default_stream=1            : 输入输出走默认流，跨 EP 拷贝不死锁
                    var cudaOpts = new OrtCUDAProviderOptions();
                    cudaOpts.UpdateOptions(new Dictionary<string, string>
                    {
                        ["device_id"] = "0",
                        ["gpu_mem_limit"] = (2L * 1024 * 1024 * 1024).ToString(),
                        ["arena_extend_strategy"] = "kSameAsRequested",
                        ["cudnn_conv_algo_search"] = "HEURISTIC",
                        ["do_copy_in_default_stream"] = "1",
                    });

                    soGpu.AppendExecutionProvider_CUDA(cudaOpts);

                    // 关掉 CPU 线程池的自旋等待。
                    //
                    // ORT 默认 session.intra_op.allow_spinning = 1：线程池在等 GPU 执行完的
                    // 空档里是**忙等**（自旋），不是休眠。实测（3060 / 16 逻辑核 / PatchCore cam1）：
                    //     默认       : 15.09 ms/张，进程 CPU 占用 49%
                    //     关自旋     : 13.43 ms/张，进程 CPU 占用  8%
                    // 也就是说 40% 的 CPU 是白烧的，而且关掉后延迟还降了 11%（少了线程唤醒竞争）。
                    // 现场表现为「GPU 明明在跑，任务管理器 CPU 却满」，会挤占相机采集和 PLC 通讯。
                    //
                    // 注意：不要再加 IntraOpNumThreads=1 —— 实测会慢到 20.92 ms，
                    // 因为留在 CPU 上的 shape/reshape 节点需要多线程。
                    soGpu.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                    soGpu.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");

                    _session = new InferenceSession(_modelPath, soGpu);
                    ProviderInfo = "CUDAExecutionProvider（显存上限 2GB）";
                }
                catch (Exception ex)
                {
                    // CUDA 不可用（缺 CUDA 运行时是最常见原因）→ 回退 CPU。
                    // 这里必须带上完整异常链：只打 ex.Message 时，"LoadLibrary failed with
                    // error 126" 完全看不出缺的是 providers_shared、providers_cuda 还是 cudnn。
                    _session = null;
                    ProviderInfo = "CUDA 不可用，已回退 CPU（" + Describe(ex) + "）";
                }
            }

            if (_session == null)
            {
                var soCpu = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };
                soCpu.AppendExecutionProvider_CPU(0);
                _session = new InferenceSession(_modelPath, soCpu);
                if (!preferGpu) ProviderInfo = "CPUExecutionProvider";
            }

            sw.Stop();
            ProviderInfo += $"  加载 {sw.ElapsedMilliseconds} ms";
        }

        /// <summary>
        /// 相对路径按 **XVision.exe 所在目录** 解析。
        /// 用 BaseDirectory 而不是 CurrentDirectory —— 后者是"启动时的工作目录"，
        /// 从快捷方式/其他目录启动时会指向别处，导致找不到模型。
        /// </summary>
        public static string ResolvePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return "";
            return System.IO.Path.IsPathRooted(path)
                ? path
                : System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, path);
        }

        /// <summary>
        /// 把异常的 InnerException 链拼成一行，便于在界面日志里一次看清根因。
        /// ORT 的 provider 加载失败会把真正原因藏在第一层之外。
        /// </summary>
        private static string Describe(Exception ex)
        {
            var parts = new List<string>();
            for (var e = ex; e != null && parts.Count < 4; e = e.InnerException)
            {
                string m = e.Message ?? e.GetType().Name;
                if (m.Length > 240) m = m.Substring(0, 240) + "...";
                parts.Add(m);
            }
            return string.Join(" <- ", parts);
        }

        /// <summary>ROI 是否已配置（四项均为 0 或范围无效时视为整图）。</summary>
        private bool HasRoi =>
            _roi.Length == 4 && _roi[2] > _roi[0] && _roi[3] > _roi[1];

        /// <summary>
        /// 推理单张图。返回分数、热图（224×224 扁平数组）和缺陷区域（原图坐标）。
        /// </summary>
        public PatchCoreResult Run(HImage image, double threshold,
                                   double blobThresholdRatio = 1.0,
                                   int minArea = 0, int maxArea = 0, int maxBlobs = 10)
        {
            if (_session == null)
                throw new InvalidOperationException("模型未加载：" + ProviderInfo);
            if (image == null || !image.IsInitialized())
                throw new ArgumentException("输入图像无效");

            var sw = Stopwatch.StartNew();
            PatchCoreResult res;

            lock (_lock)
            {
                byte[] pixels = ExtractRoi(image, out int w, out int h, out int rx, out int ry);

                var tensor = new DenseTensor<byte>(new Memory<byte>(pixels), new[] { 1, 1, h, w });
                var inputs = new[] { NamedOnnxValue.CreateFromTensor("gray", tensor) };

                float score = 0;
                float[] map = Array.Empty<float>();
                using (var results = _session.Run(inputs))
                {
                    foreach (var r in results)
                    {
                        if (r.Name == "score")
                            score = r.AsTensor<float>().ToArray()[0];
                        else if (r.Name == "anomaly_map")
                            map = r.AsTensor<float>().ToArray();
                    }
                }

                sw.Stop();
                res = new PatchCoreResult
                {
                    Score = score,
                    Map = map,
                    ElapsedMs = sw.ElapsedMilliseconds,
                    RoiWidth = w,
                    RoiHeight = h,
                    RoiX = rx,
                    RoiY = ry
                };

                double blobThr = threshold * blobThresholdRatio;
                res.Blobs = ExtractBlobs(map, blobThr, minArea, maxArea, maxBlobs);
                MapBlobsToImage(res.Blobs, w, h, rx, ry);
            }

            return res;
        }

        /// <summary>
        /// HImage → 灰度 → 按 ROI 整数裁剪 → 连续 byte 缓冲。
        /// 全程 Halcon 算子，与 YOLO 工具取像素的方式一致。
        /// </summary>
        private byte[] ExtractRoi(HImage image, out int width, out int height,
                                  out int roiX, out int roiY)
        {
            HOperatorSet.CountChannels(image, out HTuple ch);

            HImage src = image;
            HImage? gray = null;
            try
            {
                if (ch.Length > 1)
                {
                    HOperatorSet.Rgb1ToGray(image, out HObject g);
                    gray = new HImage(g);
                    src = gray;
                }

                HOperatorSet.GetImageSize(src, out HTuple iw, out HTuple ih);
                int imgW = iw.I, imgH = ih.I;

                int x0 = 0, y0 = 0, x1 = imgW, y1 = imgH;
                if (HasRoi)
                {
                    x0 = Math.Max(0, _roi[0]);
                    y0 = Math.Max(0, _roi[1]);
                    x1 = _roi[2] > x0 ? Math.Min(imgW, _roi[2]) : imgW;
                    y1 = _roi[3] > y0 ? Math.Min(imgH, _roi[3]) : imgH;
                }

                roiX = x0;
                roiY = y0;
                width = x1 - x0;
                height = y1 - y0;
                if (width <= 0 || height <= 0)
                    throw new InvalidOperationException(
                        $"ROI {x0},{y0},{x1},{y1} 与图像 {imgW}×{imgH} 无交集");

                // CropPart(row, column, width, height) —— 整数裁剪，无插值
                HImage part = src.CropPart(y0, x0, width, height);
                try
                {
                    HOperatorSet.GetImagePointer1(part, out HTuple ptr, out HTuple type,
                                                  out HTuple pw, out HTuple ph);
                    if (type.S != "byte")
                        throw new InvalidOperationException(
                            "只支持 8 位灰度图，当前图像类型为 " + type.S);

                    int len = pw.I * ph.I;
                    var buf = new byte[len];
                    Marshal.Copy((IntPtr)ptr.L, buf, 0, len);
                    return buf;
                }
                finally
                {
                    part.Dispose();
                }
            }
            finally
            {
                // 只释放我们自己创建的灰度副本；调用方的 image 不能动
                gray?.Dispose();
            }
        }

        /// <summary>
        /// 热图二值化 + 8 连通域标记，返回面积/峰值/掩码坐标的候选区域。
        /// 纯 C# 实现（224×224 只有 5 万像素，不需要 Halcon 算子）。
        /// </summary>
        private static List<PatchCoreBlob> ExtractBlobs(float[] map, double thr,
                                                       int minArea, int maxArea, int maxBlobs)
        {
            var blobs = new List<PatchCoreBlob>();
            if (map.Length != MapSize * MapSize || thr <= 0) return blobs;

            var visited = new bool[map.Length];
            var stack = new Stack<int>();
            int maxAccept = maxArea > 0 ? maxArea : int.MaxValue;

            for (int idx = 0; idx < map.Length; idx++)
            {
                if (visited[idx] || map[idx] < thr) continue;

                int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1, area = 0;
                double peak = 0;
                visited[idx] = true;
                stack.Push(idx);

                while (stack.Count > 0)
                {
                    int cur = stack.Pop();
                    int cy = cur / MapSize, cx = cur % MapSize;
                    area++;
                    if (map[cur] > peak) peak = map[cur];
                    if (cx < minX) minX = cx;
                    if (cx > maxX) maxX = cx;
                    if (cy < minY) minY = cy;
                    if (cy > maxY) maxY = cy;

                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int ny = cy + dy;
                        if (ny < 0 || ny >= MapSize) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0) continue;
                            int nx = cx + dx;
                            if (nx < 0 || nx >= MapSize) continue;
                            int ni = ny * MapSize + nx;
                            if (visited[ni] || map[ni] < thr) continue;
                            visited[ni] = true;
                            stack.Push(ni);
                        }
                    }
                }

                blobs.Add(new PatchCoreBlob
                {
                    X = minX,
                    Y = minY,
                    W = maxX - minX + 1,
                    H = maxY - minY + 1,
                    Area = area,          // 这里是热图格数，映射后再换算成原图像素²
                    Peak = peak
                });
            }

            var kept = new List<PatchCoreBlob>();
            foreach (var b in blobs)
            {
                if (b.Area < minArea) continue;
                if (b.Area > maxAccept) continue;
                kept.Add(b);
            }
            kept.Sort((a, b) => b.Area.CompareTo(a.Area));
            if (maxBlobs > 0 && kept.Count > maxBlobs) kept = kept.GetRange(0, maxBlobs);
            return kept;
        }

        /// <summary>热图坐标 → 原图坐标（计入缩放与中心裁剪偏移）。</summary>
        private static void MapBlobsToImage(List<PatchCoreBlob> blobs, int roiW, int roiH,
                                            int roiX, int roiY)
        {
            double sx = (double)roiW / ResizeTo;
            double sy = (double)roiH / ResizeTo;

            foreach (var b in blobs)
            {
                int x0 = (int)Math.Round((b.X + CropOffset) * sx);
                int y0 = (int)Math.Round((b.Y + CropOffset) * sy);
                int x1 = (int)Math.Round((b.X + b.W + CropOffset) * sx);
                int y1 = (int)Math.Round((b.Y + b.H + CropOffset) * sy);

                b.X = roiX + x0;
                b.Y = roiY + y0;
                b.W = Math.Max(1, x1 - x0);
                b.H = Math.Max(1, y1 - y0);
                b.Area = Math.Max(1, (int)Math.Round(b.Area * sx * sy));
            }
        }

        /// <summary>
        /// 置信度换算。useConfidence=false 时返回原始异常分。
        /// byMean=false: clamp(1 - score/thr, 0, 1) * 1000
        /// byMean=true : clamp((thr - score)/(thr - normalMean), 0, 1) * 1000（正常料≈1000）
        /// </summary>
        public static double ToConfidence(double score, double thr, bool byMean, double normalMean)
        {
            if (thr <= 0) return 0;
            double v;
            if (byMean)
            {
                double span = thr - normalMean;
                v = span > 1e-9 ? (thr - score) / span : 0;
            }
            else
            {
                v = 1.0 - score / thr;
            }
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            return v * 1000.0;
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _session?.Dispose();
                _session = null;
            }
        }
    }
}
