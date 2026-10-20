using System.Runtime.InteropServices;
using System.Text;
using Microsoft.ML.OnnxRuntime;

// 现场 GPU 体检工具：判断这台机器能不能让 XVision 的 ONNX 推理（YOLO / PatchCore）跑上 GPU。
// 用法: GpuProbe.exe [模型路径]     模型可省略，省略时只做环境体检。
// 把整个 publish 目录拷到目标机，双击 GpuProbe.exe 即可。

Console.OutputEncoding = Encoding.UTF8;

string model = args.Length > 0
    ? args[0]
    : @"E:\ATK\转盘机视觉\XVision\bin\Debug\net8.0-windows\PatchCore\cam1\patchcore_roi.onnx";

string baseDir = AppContext.BaseDirectory;
bool hasModel = File.Exists(model);

var report = new List<string>();
void Say(string s) { Console.WriteLine(s); report.Add(s); }
void Ok(string s) => Say("  [OK ] " + s);
void Bad(string s) => Say("  [!! ] " + s);
void Miss(string s) => Say("  [缺 ] " + s);

Say("================ GPU 体检 ================");
Say($"目录      : {baseDir}");
Say($"时间      : {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
Say($"模型      : {(hasModel ? model : "(未提供，跳过会话测试)")}");
Say("");

// ---------------------------------------------------------------- 1. 驱动
Say("== 1) NVIDIA 驱动 / CUDA driver API ==");
int driverVer = 0;
try
{
    int rc = Nv.cuInit(0);
    if (rc != 0)
        Bad($"cuInit 失败, code = {rc}（驱动异常或无可用设备）");
    else
    {
        Nv.cuDeviceGetCount(out int cnt);
        Nv.cuDriverGetVersion(out driverVer);
        string name = Nv.DeviceName(0);
        Ok($"cuInit OK / 设备数 = {cnt} / 设备 = {name}");
        Ok($"driver API 版本 = CUDA {driverVer / 1000}.{(driverVer % 1000) / 10}");
        // ORT 1.17.3 的 CUDA EP 是 CUDA 11.8 编译的，minor version compatibility 要求 >= 11.8
        int need = 11080;
        if (driverVer >= need) Ok($"驱动满足 CUDA 11.8 的最低要求 (>= 11.8)");
        else Bad($"驱动 API 只有 CUDA {driverVer / 1000}.{(driverVer % 1000) / 10}，低于 ORT 1.17.3 需要的 11.8，需升级显卡驱动");
    }
}
catch (Exception ex)
{
    Bad($"nvcuda.dll 不可用: {ex.Message}");
}
Say("");

// ---------------------------------------------------------------- 2. ORT native
Say("== 2) ONNX Runtime 文件（必须在 exe 同目录）==");
string[] ortFiles =
{
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
};
foreach (var f in ortFiles)
{
    if (File.Exists(Path.Combine(baseDir, f))) Ok(f);
    else Miss(f + "   <- 在 runtimes\\win-x64\\native 里，必须拷到本目录");
}
Say("");

// ---------------------------------------------------------------- 3. CUDA 运行时依赖
Say("== 3) provider_cuda 的 CUDA 运行时依赖（缺一个就用不了 GPU）==");
// onnxruntime_providers_cuda.dll(1.17.3) 的导入表硬依赖：CUDA 11.8 + cuDNN 8
(string Name, string From)[] cudaDeps =
{
    ("cudart64_110.dll",   "CUDA 11.8"),
    ("cublas64_11.dll",    "CUDA 11.8"),
    ("cublasLt64_11.dll",  "CUDA 11.8"),
    ("cufft64_10.dll",     "CUDA 11.8"),
    ("cudnn64_8.dll",      "cuDNN 8.x"),
};
int missing = 0;
foreach (var (name, from) in cudaDeps)
{
    // 用真实 LoadLibrary 探测：能加载才说明它在 exe 目录 / system32 / PATH 里且自身依赖齐
    IntPtr h = Nv.LoadLibraryW(Path.Combine(baseDir, name));
    if (h == IntPtr.Zero) h = Nv.LoadLibraryW(name);
    if (h != IntPtr.Zero) Ok($"{name}  ({from})");
    else { Miss($"{name}  ({from})"); missing++; }
}
Say("");

// ---------------------------------------------------------------- 4. 会话实测
Say("== 4) ORT 会话实测 ==");
if (!hasModel)
{
    Say("  跳过（未提供模型）。要测完整推理，把任一 .onnx 路径作为参数传入。");
}
else
{
    Console.WriteLine("  CPU 会话 ...");
    try
    {
        var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        so.AppendExecutionProvider_CPU(0);
        using var s = new InferenceSession(model, so);
        Console.WriteLine($"    CPU  OK  输入={string.Join(",", s.InputMetadata.Keys)} 输出={string.Join(",", s.OutputMetadata.Keys)}");
    }
    catch (Exception ex) { Console.WriteLine("    CPU  失败: " + ex.Message); }

    Console.WriteLine("  CUDA 会话 ...");
    try
    {
        using var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        so.AppendExecutionProvider_CUDA(0);
        using var s = new InferenceSession(model, so);
        Console.WriteLine("    CUDA OK  >>> 这台机器的 GPU 可用，可勾选 PatchCore 的「优先GPU」");
    }
    catch (Exception ex)
    {
        Console.WriteLine("    CUDA 失败: " + ex.Message);
        if (ex.InnerException != null) Console.WriteLine("         内层: " + ex.InnerException.Message);
    }
}
Say("");

// ---------------------------------------------------------------- 结论
Say("================ 结论 ================");
if (missing == 0 && driverVer >= 11080)
    Say("依赖看起来齐了。若上面 CUDA 会话仍失败，把完整报错发我。");
else
{
    var steps = new List<string>();
    if (missing > 0) steps.Add($"缺 {missing} 个 CUDA 运行时 dll（见第 3 节），需装 CUDA 11.8 runtime + cuDNN 8.x");
    if (driverVer > 0 && driverVer < 11080) steps.Add("显卡驱动过旧，需升级到支持 CUDA 11.8 的版本（>= 452.39）");
    Say("尚不满足 GPU 运行条件：");
    foreach (var s in steps) Say("  - " + s);
    Say("");
    Say("补齐后把 CUDA 的 5 个 dll 与两个 provider dll 一起放到本目录，再跑一次本工具。");
}

static class Nv
{
    [DllImport("nvcuda.dll")] public static extern int cuInit(int flags);
    [DllImport("nvcuda.dll")] public static extern int cuDeviceGetCount(out int count);
    [DllImport("nvcuda.dll")] public static extern int cuDriverGetVersion(out int version);
    [DllImport("nvcuda.dll", CharSet = CharSet.Ansi)]
    public static extern int cuDeviceGetName(byte[] name, int len, int dev);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr LoadLibraryW(string path);

    public static string DeviceName(int dev)
    {
        try
        {
            var buf = new byte[256];
            if (cuDeviceGetName(buf, buf.Length, dev) != 0) return "?";
            int n = Array.IndexOf(buf, (byte)0);
            return Encoding.ASCII.GetString(buf, 0, n < 0 ? buf.Length : n);
        }
        catch { return "?"; }
    }
}
