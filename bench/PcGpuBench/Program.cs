using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// PatchCore 单张推理 CPU vs CUDA 实测对比。
// 用法: PcGpuBench.exe [cam目录] [迭代次数]
//   例: PcGpuBench.exe E:\ATK\转盘机视觉\XVision\bin\Debug\net8.0-windows\PatchCore\cam1 30

Console.OutputEncoding = Encoding.UTF8;

string dir = args.Length > 0
    ? args[0]
    : @"E:\ATK\转盘机视觉\XVision\bin\Debug\net8.0-windows\PatchCore\cam1";
int iters = args.Length > 1 && int.TryParse(args[1], out var n) ? n : 30;

string modelPath = Path.Combine(dir, "patchcore_roi.onnx");
if (!File.Exists(modelPath))
{
    Console.WriteLine("找不到模型: " + modelPath);
    return 1;
}

// ROI 尺寸从 meta_v2.json 读，与现场一致
int w, h;
try
{
    using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta_v2.json")));
    var roi = doc.RootElement.GetProperty("roi");
    w = roi[2].GetInt32() - roi[0].GetInt32();
    h = roi[3].GetInt32() - roi[1].GetInt32();
}
catch
{
    w = 400; h = 400;
}

Console.WriteLine("================ PatchCore 推理基准 ================");
Console.WriteLine($"模型    : {modelPath}");
Console.WriteLine($"ROI 输入: {w} x {h}  (单通道 byte)");
Console.WriteLine($"迭代    : {iters} + 3 预热");
Console.WriteLine();

var buf = new byte[w * h];
new Random(42).NextBytes(buf);
var tensor = new DenseTensor<byte>(buf, new[] { 1, 1, h, w });

double? cpuMs = null, gpuMs = null;
string gpuName = "?";

// ------------------------------------------------------------ CPU
Console.WriteLine("-- CPU --");
try
{
    var sw = Stopwatch.StartNew();
    using var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
    so.AppendExecutionProvider_CPU(0);
    using var session = new InferenceSession(modelPath, so);
    sw.Stop();
    Console.WriteLine($"   会话创建: {sw.ElapsedMilliseconds} ms");

    cpuMs = Bench(session, w, h, buf, iters, "CPU");
}
catch (Exception ex)
{
    Console.WriteLine("   失败: " + ex.Message);
}
Console.WriteLine();

// ------------------------------------------------------------ CUDA
Console.WriteLine("-- CUDA --");
try
{
    // gpu_mem_limit 2GB：为 GTX 1650 的 4GB 显存留出显示与其他程序的余量
    var cudaOpts = new OrtCUDAProviderOptions();
    cudaOpts.UpdateOptions(new Dictionary<string, string>
    {
        ["device_id"] = "0",
        ["gpu_mem_limit"] = (2L * 1024 * 1024 * 1024).ToString(),
        ["arena_extend_strategy"] = "kSameAsRequested",
        ["cudnn_conv_algo_search"] = "HEURISTIC",
        ["do_copy_in_default_stream"] = "1",
    });

    var sw = Stopwatch.StartNew();
    using var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
    so.AppendExecutionProvider_CUDA(cudaOpts);

    // ORT 的 CPU 线程池默认 allow_spinning=1：在等 GPU 的时候忙等（自旋），
    // 表现为「GPU 推理很快但 CPU 占满」。关掉自旋，只保留计算需要的那部分。
    if (Environment.GetEnvironmentVariable("PC_NO_SPIN") == "1")
    {
        so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        so.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
        Console.WriteLine("   [实验] 已关闭 CPU 线程池自旋");
    }
    if (Environment.GetEnvironmentVariable("PC_INTRA1") == "1")
    {
        so.IntraOpNumThreads = 1;
        Console.WriteLine("   [实验] IntraOpNumThreads = 1");
    }

    using var session = new InferenceSession(modelPath, so);
    sw.Stop();
    Console.WriteLine($"   会话创建: {sw.ElapsedMilliseconds} ms");

    gpuName = QueryDeviceName();
    if (gpuName != "?") Console.WriteLine($"   设备    : {gpuName}");

    gpuMs = Bench(session, w, h, buf, iters, "CUDA");
}
catch (Exception ex)
{
    Console.WriteLine("   失败: " + ex.Message);
    if (ex.InnerException != null) Console.WriteLine("   内层: " + ex.InnerException.Message);
}
Console.WriteLine();

// ------------------------------------------------------------ 汇总
Console.WriteLine("================ 汇总 ================");
if (cpuMs.HasValue) Console.WriteLine($"CPU     : {cpuMs.Value:F2} ms/张");
else Console.WriteLine("CPU     : 不可用");
if (gpuMs.HasValue) Console.WriteLine($"CUDA    : {gpuMs.Value:F2} ms/张");
else Console.WriteLine("CUDA    : 不可用");
if (cpuMs.HasValue && gpuMs.HasValue)
{
    Console.WriteLine($"加速比  : {cpuMs.Value / gpuMs.Value:F2} x");
    Console.WriteLine();
    Console.WriteLine($"按单件节拍估算（含 YOLO 约 87ms CPU）：");
    Console.WriteLine($"  全 CPU : {cpuMs.Value + 87:F0} ms");
    Console.WriteLine($"  全 GPU : {gpuMs.Value + 20:F0} ms（YOLO 12~20ms 估）");
}
Console.WriteLine($"设备    : {gpuName}");

return 0;

// 返回中位数（毫秒）；单张推理，模拟现场调用
static double Bench(InferenceSession session, int w, int h, byte[] buf, int iters, string tag)
{
    var inputs = new[] { NamedOnnxValue.CreateFromTensor("gray", new DenseTensor<byte>(new Memory<byte>(buf), new[] { 1, 1, h, w })) };

    for (int i = 0; i < 3; i++)
    {
        using var warm = session.Run(inputs);
        foreach (var r in warm) _ = r.Name;
    }

    var proc = Process.GetCurrentProcess();
    proc.Refresh();
    var cpu0 = proc.TotalProcessorTime;
    var wall = Stopwatch.StartNew();

    var times = new List<double>(iters);
    for (int i = 0; i < iters; i++)
    {
        var sw = Stopwatch.StartNew();
        using (var res = session.Run(inputs))
        {
            foreach (var r in res) _ = r.Name;
        }
        sw.Stop();
        times.Add(sw.Elapsed.TotalMilliseconds);
    }

    wall.Stop();
    proc.Refresh();
    var cpu1 = proc.TotalProcessorTime;

    double busyMs = (cpu1 - cpu0).TotalMilliseconds;
    double util = busyMs / wall.Elapsed.TotalMilliseconds / Environment.ProcessorCount * 100.0;

    times.Sort();
    double med = times[times.Count / 2];
    Console.WriteLine($"   min {times[0]:F2} / med {med:F2} / max {times[^1]:F2} ms   (n={iters})");
    Console.WriteLine($"   进程 CPU 占用: {util:F0}%   (墙钟 {wall.ElapsedMilliseconds} ms / CPU时间 {busyMs:F0} ms / {Environment.ProcessorCount} 逻辑核)");
    return med;
}

static string QueryDeviceName()
{
    try
    {
        if (Nv.cuInit(0) != 0) return "?";
        var b = new byte[256];
        if (Nv.cuDeviceGetName(b, b.Length, 0) != 0) return "?";
        int z = Array.IndexOf(b, (byte)0);
        return Encoding.ASCII.GetString(b, 0, z < 0 ? b.Length : z);
    }
    catch { return "?"; }
}

static class Nv
{
    [DllImport("nvcuda.dll")] public static extern int cuInit(int flags);
    [DllImport("nvcuda.dll", CharSet = CharSet.Ansi)]
    public static extern int cuDeviceGetName(byte[] name, int len, int dev);
}
