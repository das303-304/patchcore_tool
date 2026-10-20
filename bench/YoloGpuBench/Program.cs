using System.Diagnostics;
using System.Text;
using cn.YoloV3;
using cn.YoloV3.Enums;
using cn.YoloV3.Models;
using SkiaSharp;

// YOLO (cn.YoloV3) CPU vs CUDA 实测。与 Yolov5Tool 用同一套 YoloOptions。
// 用法: YoloGpuBench.exe [模型路径] [迭代次数]

Console.OutputEncoding = Encoding.UTF8;

string vb = @"E:\ATK\转盘机视觉\XVision\bin\Debug\net8.0-windows";
string model = args.Length > 0 ? args[0] : Path.Combine(vb, @"yolodll\Assets\Weights\lastm.onnx");
int iters = args.Length > 1 && int.TryParse(args[1], out var n) ? n : 20;

if (!File.Exists(model)) { Console.WriteLine("找不到模型: " + model); return 1; }

Console.WriteLine("================ YOLO 推理基准 (cn.YoloV3) ================");
Console.WriteLine($"模型: {model}");
Console.WriteLine($"迭代: {iters} + 3 预热\n");

// 640×640 合成图（模型输入尺寸，用纯噪点避免全 0 被短路）
var info = new SKImageInfo(640, 640, SKColorType.Rgb888x, SKAlphaType.Opaque);
using var bmp = new SKBitmap(info);
var rnd = new Random(7);
for (int y = 0; y < 640; y++)
    for (int x = 0; x < 640; x++)
    {
        byte v = (byte)rnd.Next(60, 190);
        bmp.SetPixel(x, y, new SKColor(v, v, v));
    }
using var img = SKImage.FromBitmap(bmp);

double? cpu = null, gpu = null;

foreach (var useCuda in new[] { false, true })
{
    string tag = useCuda ? "CUDA" : "CPU";
    Console.WriteLine($"-- {tag} --");
    try
    {
        var sw = Stopwatch.StartNew();
        using var yolo = new Yolo(new YoloOptions
        {
            OnnxModel = model,
            ModelType = ModelType.ObjectDetection,
            Cuda = useCuda,
            GpuId = 0,
            PrimeGpu = false,
        });
        sw.Stop();
        Console.WriteLine($"   会话创建: {sw.ElapsedMilliseconds} ms");

        for (int i = 0; i < 3; i++) _ = yolo.RunObjectDetection(img, 0.25, 0.7);

        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        var cpu0 = proc.TotalProcessorTime;
        var wall = Stopwatch.StartNew();

        var times = new List<double>(iters);
        for (int i = 0; i < iters; i++)
        {
            var t = Stopwatch.StartNew();
            var res = yolo.RunObjectDetection(img, 0.25, 0.7);
            t.Stop();
            _ = res?.Count;
            times.Add(t.Elapsed.TotalMilliseconds);
        }

        wall.Stop();
        proc.Refresh();
        var cpu1 = proc.TotalProcessorTime;

        double util = (cpu1 - cpu0).TotalMilliseconds / wall.Elapsed.TotalMilliseconds
                      / Environment.ProcessorCount * 100.0;

        times.Sort();
        double med = times[times.Count / 2];
        Console.WriteLine($"   min {times[0]:F2} / med {med:F2} / max {times[^1]:F2} ms  (n={iters})");
        Console.WriteLine($"   进程 CPU 占用: {util:F0}%  ({(cpu1 - cpu0).TotalMilliseconds:F0} ms CPU / {Environment.ProcessorCount} 核)");
        if (useCuda) gpu = med; else cpu = med;
    }
    catch (Exception ex)
    {
        Console.WriteLine("   失败: " + ex.GetType().Name + ": " + ex.Message);
        for (var e = ex.InnerException; e != null; e = e.InnerException)
            Console.WriteLine("     内层: " + e.Message);
    }
    Console.WriteLine();
}

Console.WriteLine("================ 汇总 ================");
Console.WriteLine($"CPU  : {(cpu.HasValue ? $"{cpu.Value:F2} ms" : "失败")}");
Console.WriteLine($"CUDA : {(gpu.HasValue ? $"{gpu.Value:F2} ms" : "失败")}");
if (cpu.HasValue && gpu.HasValue)
    Console.WriteLine($"加速比: {cpu.Value / gpu.Value:F2} x");

return 0;
