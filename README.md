# PatchCore Tool

工业视觉用的 **PatchCore 无监督缺陷检测**工具集：只用 OK 图建记忆库，检测未见过的不良。
本仓库是它在「Python 建库 → ONNX 导出 → C# 产线集成」整条链路上的源码与工程记录。

> 这套代码是从产线工程 `XVision` 中抽出来的独立副本。C# 部分依赖 `Sen.Engine` 视觉框架
> （`VisionToolBase` / `VsToolGroup` / `HWndCtrl`），**不能脱离框架单独编译**，详见
> [C# 集成](#c-集成到-vision-框架)。

---

## 组成

| 目录 | 内容 |
|---|---|
| `python/` | 建库工具（带 Web UI）、ONNX 导出、检测验证、阈值标定 |
| `python/tools/` | 开发期分析脚本归档：骨干对比、kNN 基准、NG/OK 评估、报告生成 |
| `csharp/Sen.Engine/VisionTool/patchcore/` | 产线工具：引擎封装 + 工具逻辑 + 参数界面 |
| `csharp/probe_tool/` | 独立的推理探针（不依赖视觉框架，可单独跑） |
| `bench/` | 性能与 GPU 环境诊断工具（CPU/CUDA 对比、GPU 体检） |
| `docs/` | 集成方案与调参记录 |

---

## Python 建库

环境要求：Python 3.9+，PyTorch（有 GPU 更快），见 `python/requirements-frozen.txt`。

```bash
cd python
pip install -r requirements-frozen.txt

# Windows 直接双击
启动建库工具.bat        # 打开建库 Web UI（默认 http://127.0.0.1:8766）
启动检测工具.bat        # 打开检测/验证界面
```

核心脚本：

| 脚本 | 用途 |
|---|---|
| `build_tool.py` | 建库主程序：读 OK 图 → 提特征 → 建记忆库 → 导出 ONNX |
| `build_tool_ui.html` | 建库界面（单页，由 build_tool.py 提供） |
| `onnx_infer.py` | ONNX 模型推理与分数验证 |
| `inspect_tool.py` | 模型检查：骨干、库规模、阈值、meta 一致性 |
| `infer_custom.py` | 单图推理调试 |

**建库要点**

- 只用 OK 图，**不要混入 NG**，否则记忆库会被污染、NG 分数被拉平。
- 每个相机必须**各自建库**。记忆库绑定该工位的成像条件（光源/物距/ROI），
  拿 A 机的库跑 B 机必然成片误报。
- 输出两个文件，必须成对部署：
  - `patchcore_roi.onnx` —— 模型本体
  - `meta_v2.json` —— 阈值、正常均值、ROI、骨干名等元数据

---

## C# 集成到 Vision 框架

源码位置：`csharp/Sen.Engine/VisionTool/patchcore/`

| 文件 | 职责 |
|---|---|
| `PatchCoreEngine.cs` | 推理封装，与工具类解耦，可单独测试 |
| `PatchCoreModel.cs` | 参数模型（可 BinaryFormatter 序列化进工程文件） |
| `PatchCoreTool.cs` | 工具主体：取图、判定、写 PLC、显示 |
| `UcPatchCoreTool.cs` / `.Designer.cs` | 参数界面（左参数 / 右显示） |

工具通过**反射**被框架发现，依赖三个静态属性：

```csharp
public new static string GroupInfo { get; private set; } = "PatchCore";
public new static string ToolType  { get; private set; } = "PatchCore";
public new static string Explains  { get; private set; } = "无监督缺陷检测...";
```

运行画面通过 `IGroupShowObj` 接口注册：

```csharp
public string GroupShowName => ToolName + ".PatchCore";   // 例：PatchCore1.PatchCore
public void AddGroupShowOjbs(HWndCtrl mView, string ToolName) { ... }
```

> ⚠️ 注册只是第一步。**运行视图是否真的显示，取决于工位设置里的显示项勾选**
> （`VsToolGroup.ShowObjNames` 白名单）。新建工具默认不会自动加入白名单 ——
> 这是框架层的历史设计，使用时要到「工位设置 → 显示项」里手动勾选并保存工程。

### 与 YOLO 工具的语义差异

| | YOLO | PatchCore |
|---|---|---|
| 分数方向 | Score 越大越好 | 异常分**越大越差** |
| 结果形态 | 检测框 | 热图超阈值后的连通域 |
| 训练数据 | 需要标注 | 只用 OK 图 |
| 写 PLC | 原始分数 | 默认转置信度（保持 PLC 比较方向不变） |

---

## GPU 加速

推理走 ONNX Runtime **CUDA EP**。ORT 1.17.3 的 GPU 包是按 **CUDA 11.8 + cuDNN 8** 编译的，
这一点没法绕（依赖库里写死了 `_110` / `_11` ABI 后缀）。

需要放到 **exe 同目录**的文件：

```
cudart64_110.dll        cublas64_11.dll       cublasLt64_11.dll
cufft64_10.dll          cudnn64_8.dll         zlibwapi.dll
cudnn_ops_infer64_8.dll cudnn_cnn_infer64_8.dll cudnn_adv_infer64_8.dll
onnxruntime_providers_shared.dll  onnxruntime_providers_cuda.dll
```

> `onnxruntime_providers_*.dll` 在 NuGet 包里位于 `runtimes/win-x64/native/`，
> **必须提到 exe 根目录**。ORT 的 provider 加载器走 Win32 `LoadLibrary`，
> 不会去 `runtimes` 子目录找。

### 实测（RTX 3060 Laptop，单张，n=20~30 中位数）

| 模型 | CPU | CUDA | 加速 |
|---|---|---|---|
| PatchCore（ROI 1236×1249） | 60.65 ms | **14.73 ms** | 4.12 x |
| YOLO（640²，作对照） | 101.79 ms | **11.18 ms** | 9.10 x |

### 三个必须知道的坑

**1. `AppendExecutionProvider("CUDA", dict)` 不可用**

ORT 1.17.3 的通用重载只支持 QNN/SNPE/XNNPACK/AZURE，会抛：

```
Only QNN, SNPE, XNNPACK and AZURE execution providers can be enabled by this method.
```

CUDA 必须走专用重载：

```csharp
var o = new OrtCUDAProviderOptions();
o.UpdateOptions(new Dictionary<string, string> {
    ["device_id"] = "0",
    ["gpu_mem_limit"] = (2L * 1024 * 1024 * 1024).ToString(),  // 4GB 卡（如 GTX 1650）必须限
    ["arena_extend_strategy"] = "kSameAsRequested",
    ["cudnn_conv_algo_search"] = "HEURISTIC",
    ["do_copy_in_default_stream"] = "1",
});
so.AppendExecutionProvider_CUDA(o);
```

**2. 线程池自旋会把 CPU 吃满**

ORT 默认 `session.intra_op.allow_spinning = 1`，CPU 线程池在等 GPU 的空档里是**忙等**。
表现为「GPU 明明在跑，任务管理器 CPU 却 50%」。实测（16 逻辑核）：

| 配置 | 延迟 | 进程 CPU |
|---|---|---|
| 默认 | 15.09 ms | 49% |
| 关自旋 | **13.43 ms** | **8%** |
| 关自旋 + `IntraOpNumThreads=1` | 20.92 ms | 6% |

关闭方式（**不要**再叠加 `IntraOpNumThreads=1`，会慢到 20.9ms）：

```csharp
so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
so.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
```

**3. `LoadLibrary failed with error 126` 通常是依赖缺失，不是文件不存在**

可能是缺 provider 本体，也可能是它依赖的 cudnn/cublas 没到位。
用 `bench/GpuProbe` 可以逐个 dll 真实加载探测，直接点名缺哪个：

```bash
cd bench/GpuProbe && dotnet publish -c Release -r win-x64 --self-contained false -o publish
# 把 publish 整个目录拷到目标机，双击 GpuProbe.exe
```

---

## 诊断工具

| 工具 | 用途 |
|---|---|
| `bench/GpuProbe` | GPU 体检：驱动版本、provider 文件、5 个 CUDA 运行时 dll、CPU/CUDA 会话实测 |
| `bench/PcGpuBench` | PatchCore 单张延迟 CPU vs CUDA 对比，内置进程 CPU 占用统计 |
| `bench/YoloGpuBench` | YOLO 侧同样的对比（`cn.YoloV3`） |

三个工具都默认从 `XVision/bin/Debug/net8.0-windows` 引用 ORT 与模型，
换机器时改 csproj 里的 `Vb` 路径即可。

---

## 已知限制

- C# 部分依赖 `Sen.Engine` 框架，脱离框架无法编译；`csharp/probe_tool` 可独立运行。
- 模型与 `meta_v2.json` 是成对的，替换任一需同步，否则阈值/ROI 会错位。
- ORT 1.17.3 锁定 CUDA 11.8 —— 装 CUDA 12.x 无效，ABI 不兼容。
- 4GB 显存卡（GTX 1650 等）务必设置 `gpu_mem_limit`，默认 arena 会翻倍扩张。

---

## License

[Apache License 2.0](LICENSE)
