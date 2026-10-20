# PatchCore 集成到 XVision —— 实施计划（v2）

> 状态：模型已按 v2 架构重建并在独立 C# 程序里验证通过
> —— `Microsoft.ML.OnnxRuntime 1.17.3` 下 **两个相机各 8/8、分数差 0.0000**。
> 本文是实现依据，不含待验证假设。

---

## 最终决策（v7）：图像侧统一用 Halcon

用户确认「patchcore图像类也用halcon吧」。**放弃此前"零 Halcon"的方案**
（`GrayImage.cs` / `GrayImageIo.cs` 已删除），图像侧与 `Yolov5Tool` 完全一致：

| 环节 | 最终实现 |
|---|---|
| 输入图像 | `toolGroup.GetImage(ImageInSource)` → `HImage` |
| 灰度 / 裁剪 / 取像素 | `CountChannels` / `Rgb1ToGray` / `CropPart` / `GetImagePointer1` |
| 结果显示 | `HWndUnit.HWndCtrl.AddIconicVar(HImage / HRegion)` + `AddText` |
| 存图 | `HOperatorSet.WriteImage` |
| 读图（单张调试） | `HOperatorSet.ReadImage` |

**下面「零、v3 改造」及其后的 v5 / v6 章节里关于"零 Halcon / PictureBox 自绘"的描述已作废**，
保留仅作演进记录。当前生效的只有本节的描述。

### 当前文件清单（5 个）

| 文件 | 职责 |
|---|---|
| `PatchCoreModel.cs` | 参数类，`[Serializable]`，纯值类型 |
| `PatchCoreEngine.cs` | 推理封装：HImage → ROI 裁剪 → ONNX → 分数 + 热图 → 连通域 → 坐标映射 |
| `PatchCoreTool.cs` | `VisionToolBase` + `IGroupShowObj`：图像源、PLC、存图、Halcon 显示、阈值重算 |
| `UcPatchCoreTool.cs` | 界面逻辑（InitData / SaveParam / RunDone） |
| `UcPatchCoreTool.Designer.cs` | 界面布局（坐标体系与 `UcYolov5Tool.Designer.cs` 相同） |

### 多相机的模型管理（v8，解决"初建工程没有模型"）

**问题**：原来 `PatchCoreModel.ModelPath` 默认写死 `PatchCore\cam1\patchcore_roi.onnx`，
所有相机工具都指向同一个文件；而且新建工程时模型根本还不存在。

**改法**：

1. **`ModelPath` 默认留空**，留空时按**相机通道**自动解析：
   ```
   <XVision.exe 目录>\PatchCore\cam{CamChn}\patchcore_roi.onnx
   ```
   相机 1/2/3/4 各自独立，不会互相覆盖。手填绝对路径时以手填为准。

2. **`PatchCoreTool.ResolveModelPath()`** 是唯一解析入口，
   `PatchCoreEngine.ResolvePath()` 改用 **`AppDomain.CurrentDomain.BaseDirectory`**
   而不是 `Environment.CurrentDirectory` —— 后者是"启动时的工作目录"，
   从快捷方式或其他目录启动时会指向别处，导致找不到模型。

3. **界面显示模型状态**（`lblModelState`，参数页底部）：
   - 找到：`✓ 模型已找到（191 MB）  [按相机通道自动生成]` + 完整路径
   - 没找到：`✗ 模型不存在 —— 相机 2 还没建库` + 期望路径
   切换「相机通道」或改完模型路径会立即刷新。

4. **「导入模型到此相机」按钮**：初建工程的典型流程是
   Python 侧用该相机的 OK 图建库 → 得到一个 `patchcore_roi.onnx`，
   现场点这个按钮选中它，工具会自动复制到 `PatchCore\cam{N}\` 下，
   并把手填路径清空（改走自动解析）。这样多个相机的模型各自归位。

5. **缺失时日志给三步指引**：
   ```
   PatchCore 模型加载：模型文件不存在: ...
     期望路径：E://...//PatchCore//cam2//patchcore_roi.onnx
     → 每个相机必须使用自己建库的模型，不能共用（记忆库绑定该工位的成像条件）
     → 建库后把 patchcore_roi.onnx 放到上面这个路径，或在界面里改成实际路径
   ```

6. **重算阈值的错误也带指引**：模型缺失时返回期望路径；
   若所有图都读失败，把第一个错误原因带出来，并提示
   「如果提示 HALCON #2042，说明本机授权已过期」。

> **为什么每个相机必须独立模型**：记忆库存的是该工位在**特定光源/物距/ROI**下
> 提取的正常纹理特征。拿 A 机的库去跑 B 机，分数会被整体抬到缺陷量级。
> 已实测：同一批料跨相机分数放大约 1.5~1.8 倍，超过单机阈值 → 成片误报。

### 与 YOLO 的差异（只剩语义层面）

1. **分数方向相反**：YOLO 的 `Score` 越大越好；PatchCore 的异常分越大越差。
   写 PLC 时默认转成置信度（越大越好），PLC 侧比较方向不用改。
2. **缺陷来源**：YOLO 是检测框 `predictions`；PatchCore 是热图超阈值后的连通域。
3. **OK/NG 显示**：YOLO 用 `tool.IsOk` 判；PatchCore 的 `Calculate()` 恒返回 `true`
   （语义是"算法是否执行成功"），所以界面改用 `LastDefectCount > 0` 判 NG，
   否则会永远显示绿 OK。

### ⚠️ 本机无法验证 UI

本机 Halcon 授权（2026-08 评估版）已过期，`HWndUnit` 一构造就报：

```
HALCON error #2042: Feature has expired in operator set_part
```

所以 `probe_tool --layout` / `--ui` 在本机**必然失败**，这是环境问题不是代码问题。
UI 与实际推理必须在**有 Halcon 授权的机器（工控机）**上验证。

本机可以验证的部分：
- `dotnet build XVision.sln` → **0 错误**
- `probe_tool --discover` → 8 个派生类，`PatchCoreTool GroupInfo=PatchCore` 正常
- `probe_tool --halcon` → 确认当前授权状态（ReadImage / CountSeconds 都会失败）

---

## 零、v3 改造：PatchCore 不再使用 Halcon（2026-10-19）

> ⚠️ **本节及以下 v5 / v6 内容已被上面的「最终决策 v7」推翻**，仅作演进记录保留。


**动机**：本机 Halcon 授权（2026-08 评估版）已过期，任何算子都报 `#2042 Feature has expired`。
PatchCore 是新工具，不应该被 Halcon 授权绑住，所以整条链路改成纯 C#。

### 改动前后

| 环节 | v2（用 Halcon） | **v3（零 Halcon）** |
|---|---|---|
| 图像来源 | `toolGroup.GetImage(ImageInSource)` → `HImage` | **文件**：`GrayImageIo.Load(path)`（System.Drawing） |
| 灰度/通道 | `CountChannels` + `Rgb1ToGray` | 整数权重 `(77R+150G+29B)>>8` |
| ROI 裁剪 | `HImage.CropPart` | `GrayImage.Crop`（整数切片，无重采样） |
| 取像素 | `GetImagePointer1` + `Marshal.Copy` | 直接就是 `byte[]` |
| 结果提取 | 已用纯 C# 连通域 | 不变 |
| 结果显示 | `HWndCtrl.AddIconicVar(HRegion)` | **PictureBox 自绘**（ROI 绿框 / 缺陷红框 / 峰值文字） |
| 存图 | `HOperatorSet.WriteImage` | `GrayImageIo.SaveJpeg` |

`PatchCoreEngine`、`GrayImage`、`GrayImageIo` 三个类**完全不引用 HalconDotNet**。
`PatchCoreTool` 里只剩 `AddShowHObjects(HWndCtrl)` / `AddGroupShowOjbs(HWndCtrl, string)` 两个
**空实现**的签名（基类/接口强制要求），内部不调用任何算子、不添加任何 HObject。

### 新增两个文件

| 文件 | 行数 | 职责 |
|---|---|---|
| `GrayImage.cs` | 65 | `byte[] + Width/Height`，整数裁剪（越界自动钳制），含"整图不拷贝"短路 |
| `GrayImageIo.cs` | 214 | System.Drawing 读写：`Load` / `SaveJpeg` / `ToBitmap` / `FindLatest` / `ListImages` |

### 界面：复用 UcYolov5Tool 的页面结构

按用户要求，`UcPatchCoreTool` 的布局与操作流程照 `UcYolov5Tool` 抄，操作员不用重新学：

| YOLO | PatchCore |
|---|---|
| `splitContainer1`（左参数 / 右显示） | 同 |
| `tabPage1`「图像源」→ `cbxImageSource` | 同页签，内容是「图像文件夹 / 当前图片 + 两个检测按钮」 |
| `tabPage2`「查找参数」 | `tabPage2`「检测参数」：模型与阈值 / PLC 与存图 / 高级 三组 |
| `tabPage4`「测试结果」→ `listView1`（序号/字符/分数） | 同页签 → `lvBlobs`（序号/X/Y/宽/高/面积/峰值），峰值超阈值的行标红 |
| `Panel2` → `HWndUnit`（Halcon 显示） | `Panel2` → **`PictureBox` 自绘** |
| 工具条「运行/保存/启用/禁用」+ 底部日志 + 状态栏 | **继承自 `UcToolBase`，完全一致** |
| 生命周期 `InitData` → `SaveParam` → `RunDone` | 同 |

**唯一的差别是右侧显示。** `HWndUnit` 显示内容必须传 `HObject`（走 Halcon 算子），
授权过期就画不出来，所以换成 `PictureBox` 自绘，交互对齐：
**滚轮以鼠标为中心缩放、左键拖动平移、双击适应窗口**。
顶部窄条显示文件名 / 尺寸 / provider，右上角 OK / NG 色块。

### Run() 也绕开了 Halcon 计时

这是排查时发现的坑：`VisionToolBase.Run()` 内部用 `HOperatorSet.CountSeconds` 计时，
**那也是 Halcon 算子，授权过期时一样抛 #2042**——即使工具自己不用 Halcon，
只要走基类 `Run()` 就跑不起来。

`PatchCoreTool` 重写了 `Run()`，改用 `Stopwatch`：

```csharp
public override OperateResult Run(VsToolGroup toolGroup)
{
    var sw = Stopwatch.StartNew();
    var result = RunOnce(toolGroup);
    sw.Stop();
    RunTime = sw.Elapsed.TotalMilliseconds;
    toolGroup.TriggerToolStateChangeEvent(IsOk ? ToolState.Ok : ToolState.Ng, ToolName);
    TriggerToolRunDoneEvent(result);
    TriggerUpdateLogEvent(result.Message);
    TriggerUpdateWndEvent("采集图像");
    return result;
}
```

注意基类的 `ToolState` 是 `private set`，子类**无法赋值**，所以这里只发事件通知工具树刷新图标颜色。
这不影响判定：`VsToolGroup.GetIsOk()` 读的是 `tool.IsOk`，不是 `ToolState`。

> **仍然存在的框架级限制**：`VsToolGroup.RunOnce()`（整组自动运行）第 284 行也调了
> `HOperatorSet.CountSeconds`。这条路径改不掉（要动现有文件），所以**授权过期时整组运行仍会失败**。
> 单工具运行（界面点「运行」）走的是 `Tool.Run()`，已被覆盖，不受影响。

### v6：界面按 YOLO 的 Designer 体系重做

用户要求「直接复用 yolo 工具的，跳过 halcon」。现在 `UcPatchCoreTool` 拆成两个文件，
和 `UcYolov5Tool` 完全同构：

| 文件 | 说明 |
|---|---|
| `UcPatchCoreTool.Designer.cs` | **布局**，坐标体系与 `UcYolov5Tool.Designer.cs` 相同 |
| `UcPatchCoreTool.cs` | 逻辑，`InitData` / `SaveParam` / `RunDone` 写法与 `UcYolov5Tool.cs` 一致 |

**坐标体系（照抄 YOLO）**：
- 左列：Label x≈12~27，控件 x=75，宽 74（限位类 62）
- 右列：Label x≈167~202，控件 x=230/287，宽 62~74
- 行距 30px；`TabControl.ItemSize = 100×40`；字体宋体 9pt
- `splitContainer1.SplitterDistance = 391`（与 YOLO 完全相同）

**控件命名也沿用 YOLO 风格**，对照表：

| YOLO | PatchCore | 含义 |
|---|---|---|
| `label1` + `cbxImageSource` | 同 | 输入图像选择（框架图像源下拉） |
| `udCharOKScore` | `numThreshold` | 异常分阈值（**方向相反**：越大越差） |
| `udCharOKAreaLower/Upper` | `numMinArea/numMaxArea` | 面积下限/上限（中间那个 "to"） |
| `chxOKPicSave` / `chxNGPicSave` | 同 | 无缺陷/缺陷图片保存 |
| `udCharCamChn` | `numCamChn` | 相机通道 |
| `udCharCutToCamDist` | `numCutToCamDist` | 切料至相机距离 |
| `udCharPixelLocCoef` | `numCoef` | 像素-位置系数 |
| `udCharXLeftLimit/XRightLimit` | `numXLeft/numXRight` | X 左右限位 |
| `udCharYTopLimit/YBottomLimit` | `numYTop/numYBottom` | Y 顶底限位 |
| `tb_PicSaveName` | `txtPicName` | 保存名称 |
| `listView1`（序号/字符/分数） | `listView1`（序号/X/Y/宽/高/面积/峰值） | 测试结果列表 |
| `hWndShow`（HWndUnit） | **`picShow`（PictureBox）** | 唯一不同：跳过 Halcon |

PatchCore 额外增加的控件（YOLO 没有的）：模型文件 + 浏览、ROI 四个数值框、
正常样本均值、区域阈值系数、最多区域、置信度/按均值归一/Y极值/GPU 四个勾选、
「用OK图重算阈值」按钮、文件模式（图像文件夹/当前图片/两个检测按钮）。

**跳过 Halcon 的具体做法**：
- 不创建 `HWndUnit`（那是 Halcon 显示控件），右侧改为 `panelTop`（文件名 + OK/NG 色块）+ `picShow`
- `picShow` 用 `Paint` 自绘，支持**滚轮以鼠标为中心缩放 / 左键拖动平移 / 双击适应**
- `AddShowHObjects(HWndCtrl)` 空实现，`AddGroupShowOjbs` 只做 PLC 写入与存图，不碰 mView

**Designer 里最容易踩的一条**（已经踩过）：
`Controls.Add(this.splitContainer1)` 会让它落在 z-order 最底（index 最大），
而 WinForms 的 Dock **按 index 从大到小处理** → split 先占满整个区域，
盖住基类的 `toolStrip2` / `groupBox2` / `statusStrip1`。
必须在 `InitializeComponent` 末尾加：
```csharp
this.Controls.SetChildIndex(this.splitContainer1, 0);
```
同理 `tabPage4` 里要 `Controls.Add(listView1)` 再 `Controls.Add(lblSummary)`，
让 Dock=Bottom 的标签后加、先占位。

**实测（`probe_tool --layout`）**：
```
SplitContainer {0,73,984,598}   ToolStrip {0,0,984,73}
GroupBox {0,671,984,152}        StatusStrip {0,823,984,22}
SplitterDistance=427  Panel1=427  Panel2=552   （984/900 × 391，Dock 等比缩放）
图像源页：ComboBox {15,50,374,20}、两个文本框 + 浏览按钮、两个检测按钮
检测参数页：左右两列对齐，末行按钮 y=567 > ClientSize 550 → 靠 AutoScroll 滚动
```
（`tabPage2.AutoScroll = true` 是必须的：PatchCore 参数比 YOLO 多，一屏放不下）

### 图像从哪来（v5 起支持双模式）

「图像源」页有**两个单选模式**，用 `PatchCoreDataModel.UseFileImageSource` 记住选择：

**A. 框架图像源（默认，与 YOLO 完全一致）**
- 下拉选择上游工具（`toolGroup.GetImageSources`），写进 `PatchCoreTool.ImageInSource`
- 运行时 `toolGroup.GetImage(ImageInSource)` → `HImageToGray()` 拷出 8 位像素
- 这一步必须调 Halcon 算子（`CountChannels` / `Rgb1ToGray` / `GetImagePointer1`），
  **授权有效时是产线正常用法**；授权过期会返回明确错误并提示"可切到文件模式"
- 注意：`GetImage` 返回的是上游工具的 `ImageOut`，**绝对不能 Dispose**，
  否则后面的工具都拿不到图

**B. 文件（不需要 Halcon 授权）**
- `CurrentImagePath` 有值且可读 → 用它（单张复核）
- 否则 `ImageFolder` 存在 → 取目录里**最新修改**的一张（落图场景）
- 都没有 → `RunOnce` 返回失败并提示

`HImageToGray()` 是整个工具唯一调用 Halcon 算子的地方，只在模式 A 走到；
模式 B 全程 System.Drawing，零 Halcon。

### v5：界面布局的两个坑（已修）

这两条是我第一版实打实犯的错，改 UI 时务必注意：

**坑 1：`SplitterDistance` 没设 → 左栏只剩 50px**
`SplitContainer` 默认 `SplitterDistance = 50`。不显式设置的话，左侧参数区就是一条缝，
里面有控件但宽度被压到 **1px**，看起来就是"图像源页什么都没有 / 页面被遮住"。
修法：在 `SizeChanged` 里按宽度比例补偿（`ApplySplitter()`），
**且只在当前值不合理时才改**，否则会覆盖用户手动拖动的结果。

**坑 2：`Dock=Fill` 的控件必须先加、且要 `SetChildIndex(split, 0)`**
基类 `UcToolBase` 的 Designer 里 `Controls.Add` 顺序是
`toolStrip2 → groupBox2 → statusStrip1`。我们的 split 后加就在 z-order 最底部，
而 WinForms 的 Dock 是**从 z-order 最底往最前**处理的 → split 先被 Dock 成 Fill 占满全区域，
把顶部工具条（运行/保存/启用/禁用）和底部日志全盖住。
修法：`Controls.Add(split); ResumeLayout(false); Controls.SetChildIndex(split, 0);`
——**必须在 ResumeLayout 之后调**，`SuspendLayout` 期间设的 z-order 会被重排掉。

**坑 3：`Panel1MinSize` / `Panel2MinSize` 不能在构造期设**
构造时 split 宽度还是默认 150px，设 `Panel2MinSize = 220` 会立刻抛
`SplitterDistance 必须在 Panel1MinSize 和 Width - Panel2MinSize 之间`。
宽度下限改用 `ApplySplitter()` 里的常量判断。

**坑 4：`GroupBox.AutoSize` 在 `Dock=Top` 下不生效**
实测组框高度停在默认值，内容被裁掉（「X 右限位」以下全部看不见）。
改成显式 `Height`，按行数核算（每行 ≈ 30px，标题+padding ≈ 38px）。
加行时记得同步调 `NewGroup(title, height)` 的第二个参数。

**布局体检手段**：`probe_tool --layout` 把 UI 放进宿主窗口 Show 出来，
打印每个控件的实际 Bounds。实测结果：
```
SplitContainer Bounds={0,73,984,598}   ← 正确避开顶部 73px 工具条
SplitterDistance=404  Panel1=404  Panel2=576
toolStrip2  {0,0,984,73}   groupBox2 {0,671,984,152}   statusStrip1 {0,823,984,22}
```
这比肉眼截图可靠，改布局后建议跑一次。

### 验证结果（实测，原始 JPG）

`probe_tool` 直接用 `GrayImageIo` 读**原始 1800×1300 JPG**（GDI+ 解码），跑 `PatchCoreEngine`：

| 项 | 结果 |
|---|---|
| NG 10 张与 Python 分数最大偏差 | **0.0049**（阈值 0.59，判定全部一致） |
| OK 8 张抽样 | 均值 0.4953 / 最大 0.5206（Python 全量 372 张：0.4910 / 0.5675） |
| OK 误报 | **0** |
| 推理耗时 | 236 ms/张（CPU，含 GDI+ 解码约 18 ms） |
| 读写往返 | 尺寸一致；JPEG 有损最大差 15 灰阶（正常范围） |
| **UI 冒烟（构造 → InitData → Run → Dispose）** | **全通过**；NG 图 4.6122 / 1 缺陷，OK 图 0.4906 / 0 缺陷 |

> GDI+ 与 PIL 的 JPEG 解码器不同，但分数差异只有 ~0.005，远小于阈值余量（正常最大 0.5675 → 阈值 0.59）。
> 这也说明"分数对解码器细微差异不敏感"，不必担心 C# 读图和 Python 读图不一致。

### 由此解除了一个环境阻塞

原方案里 `HImage→CropPart→GetImagePointer1` 这条链路因为授权过期**无法在本机验证**。
现在这条链路不存在了，**整条 PatchCore 在无 Halcon 授权的机器上也能跑通**（已实测）。

---

## 一、工程现状（已核实）

### 工具怎么被发现

`ToolFactory.GetToolAsmList()` 反射扫描 **`Sen.Engine` 程序集**：

```csharp
foreach (Type tChild in asm.GetTypes())
    if (tChild.BaseType == (typeof(VisionToolBase)))
        toolAsmList.Add(new AssemblyData(asm, tChild));
```

**新增工具不需要注册**，放进 `Sen.Engine` 就自动出现在工具栏树（按 `GroupInfo` 分组）。

### 硬性要求（不满足会静默失败）

| 要求 | 原因 |
|---|---|
| `public PatchCoreTool(VsToolGroup toolGroup)` | `Assembly.CreateInstance(..., parameters)` 单参创建 |
| 无参构造函数 | 反序列化 |
| `public new static ToolType / GroupInfo / Explains / ScriptKeys` | 反射读静态属性 |
| `[Serializable]` | 工程用 **BinaryFormatter**（`Base/SerializationFile.cs`） |
| `InferenceSession` / `GrayImage` / `UserControl` 标 `[field: NonSerialized]` | 不可序列化 |

### 可复用能力

| 能力 | 位置 |
|---|---|
| 输入图像 / 图像源列表 | ~~`toolGroup.GetImage(ImageInSource)`~~ → v3 起改用文件（`GrayImageIo.Load`），不再依赖 HImage |
| 定位器 | `toolGroup.GetHomMat2D(HomMat2Name)` |
| 运行计时与状态 | `VisionToolBase.Run()` 已封装 `RunTime`、`ToolState`、事件 |
| UI 工具条/日志/状态栏 | `UcToolBase`（中间区域空闲，可用程序化控件 `Dock=Fill` 填充，**不必写 Designer**） |
| PLC 写入 | `ThingetPLC.WriteMultD(int coilAddress, string[] coilValue)` |
| 存图 | `ImageSaveGlobalConfig.OkImageRoot`(E:\OKDATA) / `NgImageRoot`(E:\NGDATA) + `CleanupExpiredImages()` |
| 显示 | ~~`HWndCtrl.AddIconicVar`~~ → v3 起用 `PictureBox` 自绘（零 Halcon 算子） |

### ONNX Runtime（已实测）

| 位置 | 版本 | 是否被使用 |
|---|---|---|
| `XVision/bin/Debug/net8.0-windows/onnxruntime.dll` | **1.17.3** | ✅ exe 同目录优先 |
| `XVision.deps.json` | `Microsoft.ML.OnnxRuntime.Gpu 1.17.3` | ✅ 一致 |
| `Sen.Engine/bin/.../yolodll/onnxruntime.dll` | 1.12 | ❌ 不在 XVision 输出目录 |

**模型是 opset 18，1.17.3 支持，无需升级。**
遗留：`yolodll` 里那份 1.12 建议删掉，避免单独调试 Sen.Engine 时误加载。

> 注：XVision 引用的是 Gpu 版，CUDA provider 齐全，理论上可走 GPU。
> 但 0.0000 一致性是在 **CPU EP** 下验证的，**先按 CPU 上线**，GPU 作为后续优化项。

---

## 二、v2 模型架构（ROI 由 C# 裁剪）

### 为什么改

用户要求 ROI 在新建工程时自己画。旧版把 ROI 作为常量 Slice 固化在图里，
ROI 一变就得重导模型。v2 把 ROI 裁剪交给 C#：**整数裁剪、无损、无重采样**，
不引入任何跨语言偏差。

```
C#  : HImage → CropPart(roi) → byte[]            （整数裁剪，无缩放）
ONNX: ROI图 (1,1,h,w) uint8
      → Cast/复制3通道 → Resize 256×256(antialias=1)
      → 中心裁 224 → /255 → ImageNet 归一化
      → 骨干(wideresnet50 layer2+layer3) → embedding (784,1024)
      → KNN(记忆库内嵌) → 每个 patch 的平方L2
      → ReduceMax = 图像分数；28→224 上采样 + 高斯 = 热图
输出: score float[1,1]，anomaly_map float[1,1,224,224]
```

**关键约束**：Resize 的目标尺寸必须是**常量 256×256**（再常量裁 16,16 到 224）。
ORT 1.17 不支持由算子动态计算出的 `sizes`，会报
`Number of elements of input 'sizes' (0) must be same as rank of input 'X' (4)`。

> 中间踩过一个坑：**直接缩到 224（不做 256→裁 224）会掉一档分辨率**，
> cam1 的弱缺陷 `16_` 从 0.618 掉到 0.528，检出从 10/10 变 9/10。已改回 256→裁 224。

### 模型与部署

```
<XVision.exe 目录>/
└─ PatchCore/
   ├─ cam1/  patchcore_roi.onnx (191MB)  meta_v2.json
   └─ cam2/  patchcore_roi.onnx (191MB)  meta_v2.json
```

加载约 2~4 秒 → **懒加载 + 常驻**，`InferenceSession` 标 `[field: NonSerialized]`。

---

## 三、实测指标（v2，全量标定）

| 相机 | 样本 | 均值 | std | p99 | 正常最大 | **阈值** | 最弱可检出 NG | 检出 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| cam1 | 372 | 0.4910 | 0.0167 | 0.5351 | 0.5675 | **0.59** | 0.636 | **10/10** | 204 ms |
| cam2 | 371 | 0.4604 | 0.0160 | 0.5036 | 0.5450 | **0.56** | 0.631 | **4/4** | 198 ms |

- 分离窗口：cam1 0.5675 → 0.636（**0.069**）；cam2 0.5450 → 0.631（**0.086**）
- 阈值取 `mean+6σ`，与正常最大值取较大者
- 跨语言一致性：C#(ORT 1.17.3) 与 Python **差 0.0000**（16/16 通过）

---

## 四、置信度映射（按用户指定公式）

```
conf = clamp(1 - score / 阈值, 0, 1) × 1000
```

实测落点：

| 情形 | cam1 | cam2 | conf |
|---|---|---|---|
| 正常均值 | 0.491 | 0.460 | **168 / 178** |
| 正常最大 | 0.568 | 0.545 | **38 / 27** |
| 阈值处 | 0.59 | 0.56 | **0** |
| 任意 NG | ≥0.636 | ≥0.631 | **0** |

⚠️ **注意**：这个公式下**正常料只显示 168~178 分**（不是接近 1000），
且**所有 NG 都是 0**。如果操作员把 1000 理解为"完美"，观感会很差。

若希望"正常料接近 1000"，可改成按正常均值归零：

```
conf = clamp((阈值 - score) / (阈值 - 正常均值), 0, 1) × 1000
→ 正常均值 = 1000，阈值处 = 0
```

工具里做成可切换，默认按用户指定的公式。

---

## 五、类设计（新增 6 个文件，不改任何现有文件）

`Sen.Engine/VisionTool/patchcore/`

| 文件 | 职责 |
|---|---|
| `PatchCoreModel.cs` | 参数类，`[Serializable]`，纯值类型 |
| `GrayImage.cs` | 纯 C# 灰度图（`byte[]`+宽高），整数裁剪 |
| `GrayImageIo.cs` | System.Drawing 读写图片，产出/消费 `GrayImage` |
| `PatchCoreEngine.cs` | 推理封装：加载 session、GrayImage→ROI byte[]、缺陷连通域、坐标映射 |
| `PatchCoreTool.cs` | `VisionToolBase` + `IGroupShowObj`：PLC 输出、存图（**零 Halcon**） |
| `UcPatchCoreTool.cs` | 参数界面 + PictureBox 自绘显示（程序化构建，无 Designer） |

### PatchCoreModel

```csharp
[Serializable]
public class PatchCoreModel
{
    public string ModelDir { get; set; } = "PatchCore\\cam1";
    public int[] Roi { get; set; } = { 540, 30, 1780, 1270 };  // x0,y0,x1,y1
    public double Threshold { get; set; } = 0.59;
    public double BlobThresholdRatio { get; set; } = 1.0;  // 热图二值化 = 阈值 × 该系数
    public int MinArea { get; set; } = 200;
    public int MaxArea { get; set; } = 0;                  // 0 = 不限
    public bool SaveOkImage { get; set; }
    public bool SaveNgImage { get; set; }
    public string PicSaveName { get; set; } = "";
    public int CamChn { get; set; } = 1;
    public double CutToCamDist { get; set; }
    public double PixelLocCoef { get; set; }
    public double XLeftLimit { get; set; }
    public double XRightLimit { get; set; }
    public double YTopLimit { get; set; }
    public double YBottomLimit { get; set; }
    public bool ConfidenceMode { get; set; }            // false=原始异常分, true=置信度
    public bool OutputYExtent { get; set; }             // 额外输出 Y 极值
    public int MaxBlobs { get; set; } = 10;
}
```

### 推理与坐标映射

热图 224×224 → 原图坐标（几何在模型内固定）：

```
resized 坐标 = map 坐标 + (16, 16)
ROI 坐标     = resized 坐标 × (roiW/256, roiH/256)
原图坐标     = ROI 坐标 + (x0, y0)
```

缺陷区域用 Halcon 提取：`GenImage1Const(热图)` → `Threshold(阈值)` →
`Connection` → `SelectShape(面积)` → `SmallestRectangle1` → bbox。

### PLC 输出（与 YOLO 完全一致）

```
CoilData[0] = 缺陷数 × 5
每条缺陷：X·coef·10,  (CutToCamDist+Y·coef)·10,  W·coef·10,  H·coef·10,  分数×1000
基址：CamChn 1→D2000  2→D3000  3→D4000  4→D5000
缺陷按面积从大到小排序，最多写 MaxBlobs 条
无缺陷时不写（PLC 侧「无数据 = OK」，与 YOLO 一致）
```

**可选**：勾选「输出 Y 极值」后，额外把
`Y_min、Y_max、缺陷数` 写到 **基址+100** 的 3 个寄存器，不破坏原格式。

### 判定归属（现状说明）

现有 YOLO 工具的判定链是：
`C# 按阈值/面积/位置筛掉不可信的框 → 只把幸存缺陷写 PLC → PLC 看有没有数据判 NG`。
`Calculate()` 恒返回 true，所以工具自身从不报 NG。

PatchCore 沿用同一模式，**阈值筛选在 C#，NG 裁定在 PLC**。

---

## 六、ROI 的工作流

1. 新建工程时，在界面上画 ROI（四个数值 + 图上红框，可用 Halcon 拖拽）
2. 选一批该工位的 OK 图，点「重算阈值」→ 跑模型取 `mean+6σ`
3. 保存参数

**为什么必须重算阈值**：ROI 一改，裁到的纹理就变了，分数分布跟着变，
沿用旧阈值会造成"一批好料被判 NG"且不报错。

> ROI 必须落在料面内。若含黑边/背景，分数会直接爆掉。
> 界面在设 ROI 时用亮度方差做一次粗检，提示"该区域可能含非料面内容"。

---

## 七、实施步骤与验收

| 步 | 内容 | 验收 |
|---|---|---|
| 0 | ~~ORT 版本体检~~ | ✅ 已完成：1.17.3 |
| 1 | ~~v2 模型重建 + 重标定~~ | ✅ 已完成：cam1 10/10、cam2 4/4，C# 差 0.0000 |
| 2 | `PatchCoreEngine` + 控制台验证（读 HImage、裁 ROI、缺陷框映射） | 分数与 Python 差 < 0.02；缺陷框位置肉眼正确 |
| 3 | `PatchCoreModel` + `PatchCoreTool` | 重新编译后工具树出现「PatchCore → PatchCore」，能添加、能运行 |
| 4 | `UcPatchCoreTool`（程序化 UI + 重算阈值按钮） | 参数序列化后重启仍在；重算阈值可跑 |
| 5 | 接 PLC + 存图 + Halcon 显示 | 单机空跑：寄存器数据正确，NG 图落到 `E:\NGDATA` |
| 6 | 现场实测 | 连续 1 小时：无误报、无内存增长、无句柄泄漏 |

---

## 八、风险清单

| 风险 | 影响 | 对策 |
|---|---|---|
| ROI 含黑边/背景 | 分数爆掉、成片误报 | 界面粗检提示；建议 ROI 内缩 20px |
| 191MB 模型首次加载 2~4s | 界面卡顿 | 懒加载 + 日志提示，加载后常驻 |
| `InferenceSession` 未标 NonSerialized | 保存工程抛异常 | 严格标注，`Calculate` 里判空重建 |
| HImage 是彩色 | 分数偏移（训练用灰度） | `CountChannels > 1` 时先 `Rgb1ToGray` |
| 热图二值化阈值不当 | 缺陷区域过多/过少 | `BlobThresholdRatio` 界面可调 |
| 多线程调用 RunOnce | session 并发 | 引擎内 `lock` |
| 料位漂移 | 固定 ROI 裁到料外 | 沿用 `UseHomMat2D` 定位器 |

---

## 九、工作量

| 部分 | 预估 |
|---|---|
| 模型 v2 + 标定 | ✅ 已完成 |
| PatchCoreEngine + 控制台验证 | 2~3 小时 |
| PatchCoreTool + Model | 3~4 小时 |
| UcPatchCoreTool（含重算阈值） | 3~4 小时 |
| PLC / 存图 / 联调 | 2~3 小时 |
| **剩余合计** | **约 1.5 个工作日** |

---

## 十、明确不做

- 不改 `Yolov5Tool`（新工具与之并存，可对比）
- 不改 PLC 通信层（复用 `WriteMultD`）
- 不改 `ToolFactory`（反射自动发现）
- 不动 `cn.YoloV3`
