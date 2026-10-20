using System;
using System.IO;
using System.Linq;
using Sen.Engine.VisionTool;

// PatchCore 工具的离线探针：
//   --layout   把 UI 放进宿主窗口 Show，打印每个控件实际位置（查遮挡/裁切）
//   --ui       构造 Tool + UI，走 InitData → Run → Dispose（需要 Halcon 授权）
//   --halcon   探测本机 Halcon 算子在授权过期后还能不能调
//   --discover 模拟 ToolFactory 的反射发现
internal static class Program
{
    private const string Model = @"C:\Users\Administrator\Desktop\PatchCore\models_onnx\皮料相机1\patchcore_roi.onnx";
    private const string OkDir = @"C:\Users\Administrator\Desktop\2026-10-08\2026-10-08\皮料相机1";
    private const string NgDir = @"C:\Users\Administrator\Desktop\NG\CAM1";

    private static readonly int[] Roi = { 540, 30, 1780, 1270 };

    [STAThread]
    private static int Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        string mode = args.Length > 0 ? args[0] : "--layout";

        switch (mode)
        {
            case "--ui": return UiSmokeTest();
            case "--halcon": return ProbeHalcon();
            case "--discover": return Discover();
            case "--paths": return ShowPaths();
            case "--meta": return CheckMeta();
            case "--serial": return CheckSerial();
            default: return LayoutCheck();
        }
    }

    // ==================================================================
    // 序列化往返：确认参数模型的 bool 能不能存进工程文件
    // ==================================================================
    private static int CheckSerial()
    {
        var tool = new PatchCoreTool { ToolName = "serial-test" };
        var m = tool.PatchCoreDataModel;
        m.UseConfidence = true;
        m.ConfidenceByMean = true;
        m.OutputYExtent = true;
        m.PreferGpu = true;
        m.SaveOkImage = true;
        m.SaveNgImage = true;
        m.Threshold = 1.23;
        m.CamChn = 2;

        var bf = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
        using var ms = new System.IO.MemoryStream();
        bf.Serialize(ms, tool);
        ms.Position = 0;
        var t2 = (PatchCoreTool)bf.Deserialize(ms);
        var r = t2.PatchCoreDataModel;

        Console.WriteLine("序列化 -> 反序列化：");
        Console.WriteLine($"  UseConfidence    = {r.UseConfidence}");
        Console.WriteLine($"  ConfidenceByMean = {r.ConfidenceByMean}");
        Console.WriteLine($"  OutputYExtent    = {r.OutputYExtent}");
        Console.WriteLine($"  PreferGpu        = {r.PreferGpu}");
        Console.WriteLine($"  SaveOkImage      = {r.SaveOkImage}");
        Console.WriteLine($"  SaveNgImage      = {r.SaveNgImage}");
        Console.WriteLine($"  Threshold        = {r.Threshold}");
        Console.WriteLine($"  CamChn           = {r.CamChn}");

        bool ok = r.UseConfidence && r.ConfidenceByMean && r.OutputYExtent
                  && r.PreferGpu && r.SaveOkImage && r.SaveNgImage
                  && Math.Abs(r.Threshold - 1.23) < 1e-9 && r.CamChn == 2;
        Console.WriteLine(ok ? "=> model 序列化正常，问题在 UI 流程" : "=> model 序列化丢值");
        return ok ? 0 : 1;
    }

    // ==================================================================
    // 元数据检查：验证 ReadBuildMeta 能读到 models 旁边的 meta_v2.json
    // ==================================================================
    private static int CheckMeta()
    {
        const string deploy = @"E://ATK//转盘机视觉//XVision//bin//Debug//net8.0-windows//PatchCore//cam1//patchcore_roi.onnx";
        var tool = new PatchCoreTool { ToolName = "PatchCore1" };
        tool.PatchCoreDataModel.CamChn = 1;
        tool.PatchCoreDataModel.ModelPath = deploy;

        Console.WriteLine("模型路径：" + tool.ResolveModelPath());
        Console.WriteLine("模型存在：" + tool.ModelExists);

        var meta = tool.ReadBuildMeta();
        if (meta == null)
        {
            Console.WriteLine("✗ ReadBuildMeta 返回 null（没找到 meta_v2.json 或没有 threshold）");
            return 1;
        }

        Console.WriteLine($"✓ 阈值 {meta.Threshold:F2}   记忆库 {meta.NBank} 向量");
        Console.WriteLine($"  留出 n={meta.HoldN} mean={meta.HoldMean:F4} max={meta.HoldMax:F4}");
        Console.WriteLine("  ROI " + (meta.Roi == null ? "(文件里没有)" : string.Join(",", meta.Roi)));
        Console.WriteLine("  Stamp " + meta.Stamp);
        return 0;
    }

    // ==================================================================
    // 布局体检
    // ==================================================================
    private static int LayoutCheck()
    {
        System.Windows.Forms.Application.EnableVisualStyles();

        var tool = new PatchCoreTool { ToolName = "PatchCore1" };
        tool.PatchCoreDataModel.ModelPath = Model;

        System.Windows.Forms.UserControl ui;
        try
        {
            ui = new UcPatchCoreTool(tool, null);
        }
        catch (Exception ex)
        {
            Console.WriteLine("构造 UI 失败：" + ex.GetType().Name + " / " + ex.Message);
            Console.WriteLine("（HWndUnit 是 Halcon 控件，本机授权过期时会在这里失败）");
            return 1;
        }

        var form = new System.Windows.Forms.Form
        {
            Text = "layout check",
            Size = new System.Drawing.Size(1000, 900),
            StartPosition = System.Windows.Forms.FormStartPosition.Manual,
            Location = new System.Drawing.Point(-2000, -2000),
            ShowInTaskbar = false
        };
        form.Controls.Add(ui);
        ui.Dock = System.Windows.Forms.DockStyle.Fill;

        form.Show();
        System.Windows.Forms.Application.DoEvents();

        Console.WriteLine($"宿主窗口 {form.ClientSize.Width}×{form.ClientSize.Height}\n");

        Console.WriteLine("── UcToolBase 自带的三个控件（不能被 split 盖住）──");
        foreach (System.Windows.Forms.Control c in ui.Controls)
        {
            Console.WriteLine($"  {c.GetType().Name,-16} {c.Name,-18} " +
                              $"Bounds={c.Bounds}  Dock={c.Dock}");
        }

        var split = ui.Controls.OfType<System.Windows.Forms.SplitContainer>().FirstOrDefault();
        if (split != null)
        {
            Console.WriteLine($"\n── 分栏 ──  SplitterDistance={split.SplitterDistance}  " +
                              $"Panel1={split.Panel1.Width}  Panel2={split.Panel2.Width}");
        }

        var tabs = split?.Panel1.Controls.OfType<System.Windows.Forms.TabControl>().FirstOrDefault();
        if (tabs != null)
        {
            Console.WriteLine($"\n── 页签 ──  Bounds={tabs.Bounds}  共 {tabs.TabPages.Count} 页");
            foreach (System.Windows.Forms.TabPage tp in tabs.TabPages)
            {
                Console.WriteLine($"  「{tp.Text}」 ClientSize={tp.ClientSize}  控件 {tp.Controls.Count} 个");
                DumpChildren(tp, 6);
            }
        }

        var show = split?.Panel2.Controls.OfType<Sen.Engine.ImageControl.HWndUnit>().FirstOrDefault();
        Console.WriteLine(show != null
            ? $"\n── 右侧显示 ── HWndUnit Bounds={show.Bounds}（Halcon 控件，与 YOLO 一致）"
            : "\n── 右侧显示 ── 未找到 HWndUnit");

        form.Close();
        ui.Dispose();
        tool.Dispose();
        System.Windows.Forms.Application.DoEvents();
        return 0;
    }

    private static void DumpChildren(System.Windows.Forms.Control parent, int indent, int depth = 0)
    {
        if (depth > 2) return;
        string pad = new string(' ', indent);
        foreach (System.Windows.Forms.Control c in parent.Controls)
        {
            string extra = "";
            if (c is System.Windows.Forms.Label lb) extra = $"  「{Trim(lb.Text)}」";
            if (c is System.Windows.Forms.Button bt) extra = $"  [{bt.Text}]";
            Console.WriteLine($"{pad}- {c.GetType().Name,-20} {c.Bounds}{extra}");
            DumpChildren(c, indent + 2, depth + 1);
        }
    }

    private static string Trim(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        s = s.Replace("\r", " ").Replace("\n", " ");
        return s.Length > 20 ? s.Substring(0, 20) + "…" : s;
    }

    // ==================================================================
    // UI 冒烟（需要 Halcon 授权）
    // ==================================================================
    private static int UiSmokeTest()
    {
        System.Windows.Forms.Application.EnableVisualStyles();

        Console.WriteLine("1) 构造 PatchCoreTool ...");
        var tool = new PatchCoreTool { ToolName = "PatchCore1" };
        tool.PatchCoreDataModel.ModelPath = Model;
        tool.PatchCoreDataModel.Roi = (int[])Roi.Clone();
        tool.PatchCoreDataModel.Threshold = 0.59;
        tool.PatchCoreDataModel.NormalMean = 0.4910;
        Console.WriteLine("    OK");

        Console.WriteLine("2) 构造 UcPatchCoreTool（含 HWndUnit）...");
        System.Windows.Forms.UserControl ui;
        try
        {
            ui = new UcPatchCoreTool(tool, null);
            Console.WriteLine($"    OK  {ui.Width}×{ui.Height}");
        }
        catch (Exception ex)
        {
            Console.WriteLine("    失败：" + ex.GetType().Name + " / " + ex.Message);
            Console.WriteLine("    HWndUnit 初始化需要 Halcon，本机授权过期时必然失败——属于预期。");
            return 1;
        }

        Console.WriteLine("3) InitData ...");
        ((UcPatchCoreTool)ui).InitData();
        Console.WriteLine("    OK");

        // 「打开图片并检测」已删除，单张调试改由上游图像源工具提供图像。
        Console.WriteLine("4) 跳过 Run：工具已不含手动读图入口，需由图像源工具供图");

        Console.WriteLine("5) 释放 ...");
        ui.Dispose();
        tool.Dispose();
        Console.WriteLine("    OK");
        return 0;
    }

    // ==================================================================
    // Halcon 算子可用性
    // ==================================================================
    private static int ProbeHalcon()
    {
        HalconDotNet.HTuple s;
        try
        {
            HalconDotNet.HOperatorSet.CountSeconds(out s);
            Console.WriteLine($"CountSeconds : 可用 ({s.D:F3})");
        }
        catch (Exception ex)
        {
            Console.WriteLine("CountSeconds : 失败  " + ex.Message);
        }

        try
        {
            HalconDotNet.HOperatorSet.ReadImage(out HalconDotNet.HObject obj, Model);
            Console.WriteLine("ReadImage    : 可用");
            obj.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine("ReadImage    : 失败  " + ex.Message);
        }

        return 0;
    }

    /// <summary>多相机模型路径解析检查（不需要 Halcon）。</summary>
    private static int ShowPaths()
    {
        Console.WriteLine("ModelPath 留空时，按相机通道自动生成：\n");
        for (int cam = 1; cam <= 4; cam++)
        {
            var t = new PatchCoreTool { ToolName = "PatchCore" + cam };
            t.PatchCoreDataModel.CamChn = cam;
            Console.WriteLine($"  CamChn={cam}  →  {t.ResolveModelPath()}");
            Console.WriteLine($"            存在={t.ModelExists}");
            t.Dispose();
        }

        Console.WriteLine("\n手动指定绝对路径时以手填为准：");
        var t2 = new PatchCoreTool { ToolName = "PatchCore9" };
        t2.PatchCoreDataModel.CamChn = 3;
        t2.PatchCoreDataModel.ModelPath = @"D://models//other.onnx";
        Console.WriteLine($"  →  {t2.ResolveModelPath()}");
        t2.Dispose();
        return 0;
    }

    // ==================================================================
    // 反射发现（模拟 ToolFactory）
    // ==================================================================
    private static int Discover()
    {
        var asm = typeof(PatchCoreEngine).Assembly;
        var baseType = asm.GetType("Sen.Engine.VisionTool.VisionToolBase");
        if (baseType == null) { Console.WriteLine("找不到 VisionToolBase"); return 2; }

        int n = 0;
        foreach (var t in asm.GetTypes())
        {
            if (t.IsAbstract || !baseType.IsAssignableFrom(t) || t == baseType) continue;
            n++;

            var gi = t.GetProperty("GroupInfo",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            var tt = t.GetProperty("ToolType",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);

            string group = gi?.GetValue(null) as string ?? "";
            string type = tt?.GetValue(null) as string ?? "";
            string mark = t.Name == "PatchCoreTool" ? "   <== 新增工具" : "";
            Console.WriteLine($"{t.FullName,-52} GroupInfo={group,-14} ToolType={type}{mark}");
        }
        Console.WriteLine($"\n共发现 {n} 个 VisionToolBase 派生类。");
        return 0;
    }
}
