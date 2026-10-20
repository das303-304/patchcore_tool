using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Sen.Communication;
using Sen.Engine.ImageControl;

namespace Sen.Engine.VisionTool
{
    /// <summary>
    /// PatchCore 工具的参数界面。
    ///
    /// **与 UcYolov5Tool 完全同构**（布局见 UcPatchCoreTool.Designer.cs）：
    ///   splitContainer1（左参数 / 右显示）
    ///     ├─ Panel1: tabControl → tabPage1「图像源」/ tabPage2「检测参数」/ tabPage4「测试结果」
    ///     └─ Panel2: hWndShow（HWndUnit，Halcon 显示控件）
    ///   顶部工具条、底部日志、状态栏由基类 UcToolBase 提供；
    ///   生命周期 InitData → SaveParam → RunDone 与 YOLO 完全相同。
    ///
    /// 图像侧全部走 Halcon：显示用 HWndCtrl.AddIconicVar(HImage/HRegion)，
    /// 取像素用 GetImagePointer1，读图 ReadImage，存图 WriteImage。
    /// </summary>
    public partial class UcPatchCoreTool : UcToolBase
    {
        private HWndCtrl mView;
        private PatchCoreTool tool;
        private VsToolGroup toolGroup;

        public UcPatchCoreTool(PatchCoreTool tool, VsToolGroup toolGroup)
        {
            InitializeComponent();

            mView = this.hWndShow.HWndCtrl;     // hWndShow 是 HWndUnit 类，HWndCtrl 是其属性
            this.tool = tool;
            this.toolGroup = toolGroup;
            this.hWndShow.CameraMessage = this.tool.ToolName;

            BindEvents();
            base.Init(tool, toolGroup);
        }

        private void BindEvents()
        {
            btnModel.Click += (s, e) =>
            {
                using var dlg = new OpenFileDialog
                {
                    Filter = "ONNX 模型|*.onnx|所有文件|*.*",
                    CheckFileExists = true
                };
                if (dlg.ShowDialog() == DialogResult.OK) txtModel.Text = dlg.FileName;
            };

            // 相机通道变了 → 自动模型路径也跟着变，状态要刷新
            numCamChn.ValueChanged += (s, e) => { SaveParam(); UpdateModelState(); };
            // 手动改完模型路径后刷新状态
            txtModel.Leave += (s, e) => { SaveParam(); UpdateModelState(); };

            btnImport.Click += (s, e) => ImportModel();
        }

        /// <summary>
        /// 尺寸确定后再校正一次分栏位置。Designer 里已设 391（与 YOLO 相同），
        /// 宿主窗口很窄时 Dock 缩放可能压得不合理，这里兜一下。
        /// </summary>
        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            try
            {
                int width = splitContainer1.Width;
                const int minLeft = 360, minRight = 200;
                int max = width - minRight - splitContainer1.SplitterWidth;
                if (max > minLeft && splitContainer1.SplitterDistance < minLeft)
                    splitContainer1.SplitterDistance = minLeft;
            }
            catch { /* 尺寸未定，忽略 */ }
        }

        /// <summary>
        /// InitData 执行期间为 true。
        /// 那时控件还没赋完值，此时触发的 ValueChanged（如 numCamChn）会走到
        /// SaveParam，把「还没初始化的复选框/文本框」的默认值写回参数，
        /// 导致保存后重新打开工程，勾选项全部丢失。
        /// </summary>
        private bool loading;

        // ==================================================================
        // 生命周期（与 UcYov5Tool 一致）
        // ==================================================================
        public override void InitData()
        {
            loading = true;
            try
            {
                var m = tool.PatchCoreDataModel;

                // 换过模型就先把检测参数同步成模型自带的，再刷控件
                SyncParamsFromMeta();

                // ---- 图像源（与 YOLO 完全一样：下拉选上游工具）----
                if (toolGroup != null)
                {
                    try
                    {
                        cbxImageSource.DataSource = toolGroup.GetImageSources(tool.ToolName);
                        if (!string.IsNullOrEmpty(tool.ImageInSource))
                            cbxImageSource.SelectedItem = tool.ImageInSource;
                        else
                            cbxImageSource.SelectedIndex = -1;
                    }
                    catch { /* 无工具组上下文 */ }
                }

                // ---- 检测参数 ----
                txtModel.Text = m.ModelPath;

                RefreshDetectionParams();     // ROI / 阈值 / 正常均值
                numBlobRatio.Value = Clamp((decimal)m.BlobThresholdRatio, numBlobRatio.Minimum, numBlobRatio.Maximum);
                numMinArea.Value = Clamp(m.MinArea, numMinArea.Minimum, numMinArea.Maximum);
                numMaxArea.Value = Clamp(m.MaxArea, numMaxArea.Minimum, numMaxArea.Maximum);
                numMaxBlobs.Value = Clamp(m.MaxBlobs, numMaxBlobs.Minimum, numMaxBlobs.Maximum);
                numCamChn.Value = Clamp(m.CamChn, numCamChn.Minimum, numCamChn.Maximum);
                numCutToCamDist.Value = Clamp((decimal)m.CutToCamDist, numCutToCamDist.Minimum, numCutToCamDist.Maximum);
                numCoef.Value = Clamp((decimal)m.PixelLocCoef, numCoef.Minimum, numCoef.Maximum);
                numXLeft.Value = Clamp((decimal)m.XLeftLimit, numXLeft.Minimum, numXLeft.Maximum);
                numXRight.Value = Clamp((decimal)m.XRightLimit, numXRight.Minimum, numXRight.Maximum);
                numYTop.Value = Clamp((decimal)m.YTopLimit, numYTop.Minimum, numYTop.Maximum);
                numYBottom.Value = Clamp((decimal)m.YBottomLimit, numYBottom.Minimum, numYBottom.Maximum);

                txtPicName.Text = m.PicSaveName;
                chxOKPicSave.Checked = m.SaveOkImage;
                chxNGPicSave.Checked = m.SaveNgImage;
                chxConfidence.Checked = m.UseConfidence;
                chxConfByMean.Checked = m.ConfidenceByMean;
                chxYExtent.Checked = m.OutputYExtent;
                chxGPU.Checked = m.PreferGpu;

                UpdateModelState();
                RefreshResultView();
        
            }
            finally
            {
                loading = false;
            }
        }

        //保存参数
        public override void SaveParam()
        {
            // 初始化期间控件值还不完整，回写会把参数写成控件的默认值
            if (loading) return;

            var m = tool.PatchCoreDataModel;

            m.ModelPath = txtModel.Text.Trim();
            m.Roi = new[]
            {
                (int)numRoi0.Value, (int)numRoi1.Value,
                (int)numRoi2.Value, (int)numRoi3.Value
            };

            m.Threshold = (double)numThreshold.Value;
            m.NormalMean = (double)numNormalMean.Value;
            m.BlobThresholdRatio = (double)numBlobRatio.Value;
            m.MinArea = (int)numMinArea.Value;
            m.MaxArea = (int)numMaxArea.Value;
            m.MaxBlobs = (int)numMaxBlobs.Value;
            m.CamChn = (int)numCamChn.Value;
            m.CutToCamDist = (double)numCutToCamDist.Value;
            m.PixelLocCoef = (double)numCoef.Value;
            m.XLeftLimit = (double)numXLeft.Value;
            m.XRightLimit = (double)numXRight.Value;
            m.YTopLimit = (double)numYTop.Value;
            m.YBottomLimit = (double)numYBottom.Value;

            m.PicSaveName = txtPicName.Text.Trim();
            m.SaveOkImage = chxOKPicSave.Checked;
            m.SaveNgImage = chxNGPicSave.Checked;
            m.UseConfidence = chxConfidence.Checked;
            m.ConfidenceByMean = chxConfByMean.Checked;
            m.OutputYExtent = chxYExtent.Checked;
            m.PreferGpu = chxGPU.Checked;

            if (cbxImageSource.SelectedIndex != -1)
                tool.ImageInSource = cbxImageSource.SelectedValue?.ToString() ?? "";
        }

        protected override void RunDone(object sender, OperateResult e)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action(() => RunDone(sender, e)));
                return;
            }

            mView.ClearList();
            tool.AddShowHObjects(mView);
            mView.Repaint();

            // 与 YOLO 一致：绿 OK / 红 NG。
            // 注意这里用缺陷数判，而不是 IsOk —— PatchCore 的 Calculate() 恒返回 true
            // （语义是"算法是否执行成功"），用 IsOk 会永远显示 OK。
            if (tool.LastDefectCount > 0) mView.ShowNg();
            else mView.ShowOK();

            RefreshResultView();
        }

        /// <summary>
        /// 换模型后把检测参数同步成模型自带的（ROI / 阈值 / 正常均值）。
        ///
        /// 只在模型换过时执行一次：靠 PatchCoreModel.MetaStamp 记下已应用的
        /// meta_v2.json 版本（文件写入时间），之后用户在界面上调参不会被覆盖，
        /// 下次导入/替换模型又会自动重新同步。
        ///
        /// ROI 必须同步 —— 模型是用建库时的 ROI 裁出的记忆库建成的，
        /// C# 这边若沿用旧 ROI，裁的不是同一块料面，分数会整体漂移。
        /// </summary>
        private void SyncParamsFromMeta()
        {
            var m = tool.PatchCoreDataModel;
            var meta = tool.ReadBuildMeta();
            if (meta == null) return;
            if (m.MetaStamp == meta.Stamp) return;      // 这个模型已经同步过

            var changed = new List<string>();

            if (meta.Roi != null)
            {
                var cur = m.RoiCopy();
                if (cur[0] != meta.Roi[0] || cur[1] != meta.Roi[1]
                    || cur[2] != meta.Roi[2] || cur[3] != meta.Roi[3])
                {
                    m.Roi = new[] { meta.Roi[0], meta.Roi[1], meta.Roi[2], meta.Roi[3] };
                    changed.Add($"ROI [{meta.Roi[0]},{meta.Roi[1]},{meta.Roi[2]},{meta.Roi[3]}]");
                }
            }

            if (Math.Abs(m.Threshold - meta.Threshold) > 0.0005)
            {
                m.Threshold = meta.Threshold;
                changed.Add($"阈值 {meta.Threshold:F2}");
            }

            if (meta.HoldMean > 0 && Math.Abs(m.NormalMean - meta.HoldMean) > 0.0005)
            {
                m.NormalMean = Math.Round(meta.HoldMean, 4);
                changed.Add($"正常均值 {m.NormalMean:F4}");
            }

            m.MetaStamp = meta.Stamp;

            if (changed.Count > 0)
            {
                try
                {
                    tool.TriggerUpdateLogEvent("已按模型 meta_v2.json 同步：" + string.Join("，", changed)
                                               + $"（记忆库 {meta.NBank} 向量）");
                }
                catch { /* 日志控件还没准备好 */ }
            }
        }

        private static decimal Clamp(decimal v, decimal lo, decimal hi)
        {
            if (v < lo) return lo;
            if (v > hi) return hi;
            return v;
        }

        /// <summary>把 ROI / 阈值 / 正常均值从参数对象刷到控件上。</summary>
        private void RefreshDetectionParams()
        {
            var m = tool.PatchCoreDataModel;
            var roi = m.RoiCopy();
            numRoi0.Value = Clamp(roi[0], numRoi0.Minimum, numRoi0.Maximum);
            numRoi1.Value = Clamp(roi[1], numRoi1.Minimum, numRoi1.Maximum);
            numRoi2.Value = Clamp(roi[2], numRoi2.Minimum, numRoi2.Maximum);
            numRoi3.Value = Clamp(roi[3], numRoi3.Minimum, numRoi3.Maximum);
            numThreshold.Value = Clamp((decimal)m.Threshold, numThreshold.Minimum, numThreshold.Maximum);
            numNormalMean.Value = Clamp((decimal)m.NormalMean, numNormalMean.Minimum, numNormalMean.Maximum);
        }

        /// <summary>
        /// 把已建好的模型导入到本相机的标准位置（PatchCore\cam{N}\patchcore_roi.onnx）。
        ///
        /// 初建工程的典型流程：Python 侧用该相机的 OK 图建库 → 得到一个 patchcore_roi.onnx
        /// → 在这里点「导入模型到此相机」，文件会自动落到本相机目录，
        /// 路径参数清空后按相机通道自动解析（不会和别的相机互相覆盖）。
        /// </summary>
        private void ImportModel()
        {
            using var dlg = new OpenFileDialog
            {
                Filter = "ONNX 模型|*.onnx|所有文件|*.*",
                CheckFileExists = true,
                Title = "选择为该相机建库得到的 patchcore_roi.onnx"
            };
            if (dlg.ShowDialog() != DialogResult.OK) return;

            string dest;
            bool copiedMeta = false;      // 供 try 之后的提示使用
            try
            {
                dest = tool.ResolveModelPath();

                string dir = Path.GetDirectoryName(dest);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                if (string.Equals(Path.GetFullPath(dlg.FileName),
                                  Path.GetFullPath(dest), StringComparison.OrdinalIgnoreCase))
                {
                    // 本身就是目标文件，不用拷
                }
                else
                {
                    File.Copy(dlg.FileName, dest, true);
                }

                // 建库工具会在 onnx 旁边放 meta_v2.json（ROI / 阈值 / 留出分布）。
                // 只拷 onnx 的话，参数就只剩旧工程里的值 —— 那是最容易出错的地方。
                try
                {
                    string srcMeta = Path.Combine(Path.GetDirectoryName(dlg.FileName) ?? "", "meta_v2.json");
                    string dstDir = Path.GetDirectoryName(dest) ?? "";
                    if (File.Exists(srcMeta) && dstDir.Length > 0)
                    {
                        File.Copy(srcMeta, Path.Combine(dstDir, "meta_v2.json"), true);
                        copiedMeta = true;
                    }
                }
                catch { /* meta 拷不过去不影响 onnx 本身 */ }

                // 清空手填路径 → 改用按相机通道自动解析
                tool.PatchCoreDataModel.ModelPath = "";
                txtModel.Text = "";
                SaveParam();

                tool.ResetEngine();
                SyncParamsFromMeta();       // 换模型了 → 按新模型的 meta 更新参数
                RefreshDetectionParams();
                UpdateModelState();
            }
            catch (Exception ex)
            {
                MessageBox.Show("导入失败：" + ex.Message, "导入模型",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            MessageBox.Show($"已导入到：\r\n{dest}\r\n\r\n" +
                            (copiedMeta
                                ? "已同时导入 meta_v2.json，ROI / 阈值已按模型更新。\r\n"
                                : "⚠ 同目录下没有 meta_v2.json，ROI / 阈值保持原样，请手动核对。\r\n") +
                            $"该模型只适用于相机 {tool.PatchCoreDataModel.CamChn}。",
                            "导入模型", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        /// <summary>
        /// 显示模型状态。
        /// ModelPath 为空时按相机通道自动生成路径 —— 新建工程时不用手填，
        /// 但每个相机必须放自己的模型（记忆库绑定该工位的成像条件）。
        /// </summary>
        private void UpdateModelState()
        {
            if (lblModelState == null) return;

            var m = tool.PatchCoreDataModel;
            string path = tool.ResolveModelPath();
            bool auto = string.IsNullOrWhiteSpace(m.ModelPath);

            if (tool.ModelExists)
            {
                long mb = 0;
                try { mb = new FileInfo(path).Length / 1024 / 1024; } catch { }

                // 建库工具会在模型旁留 meta_v2.json（含留出标定出的阈值）。
                // 阈值不能用建库图自评——特征在库里、分数被压低，只有「未入库图」
                // 的分布才是真上界，所以这里直接引用建库结论，不再本地重算。
                string line2;
                bool mismatch = false;
                if (tool.TryReadBuildMeta(out double thr, out double hmax, out int hn))
                {
                    mismatch = Math.Abs(m.Threshold - thr) > 0.005;
                    line2 = $"建库建议阈值 {thr:F2}（留出 {hn} 张 / 最大 {hmax:F3}）" +
                            (mismatch ? $"　当前 {m.Threshold:F2} ✗" : "　✓ 已一致");
                }
                else
                {
                    line2 = $"未找到 meta_v2.json，阈值需按建库结果手填（当前 {m.Threshold:F2}）";
                }

                lblModelState.ForeColor = mismatch ? Color.DarkOrange : Color.ForestGreen;
                lblModelState.Text = $"✓ 模型已找到（{mb} MB）" +
                                     (auto ? "  [按相机通道自动生成]" : "") + "\r\n" +
                                     line2 + "\r\n" + path;
            }
            else
            {
                lblModelState.ForeColor = Color.Firebrick;
                lblModelState.Text = $"✗ 模型不存在 —— 相机 {m.CamChn} 还没建库" +
                                     (auto ? "  [按相机通道自动生成]" : "") + "\r\n" + path;
            }
        }

        // ==================================================================
        // 测试结果列表
        // ==================================================================
        private void RefreshResultView()
        {
            if (IsDisposed) return;

            var m = tool.PatchCoreDataModel;
            var blobs = tool.KeptBlobs;

            listView1.BeginUpdate();
            listView1.Items.Clear();
            int i = 1;
            foreach (var b in blobs)
            {
                var it = new ListViewItem(i.ToString());
                it.SubItems.Add(b.X.ToString());
                it.SubItems.Add(b.Y.ToString());
                it.SubItems.Add(b.W.ToString());
                it.SubItems.Add(b.H.ToString());
                it.SubItems.Add(b.Area.ToString());
                it.SubItems.Add(b.Peak.ToString("F3"));
                it.ForeColor = b.Peak >= m.Threshold ? Color.Firebrick : Color.Black;
                listView1.Items.Add(it);
                i++;
            }
            listView1.EndUpdate();

            double score = tool.LastScore;
            lblSummary.Text = blobs.Count == 0 && score <= 0
                ? "尚未运行"
                : $"异常分 {score:F3} / 阈值 {m.Threshold:F2}    缺陷 {blobs.Count} 个    " +
                  $"{tool.LastResult?.ElapsedMs ?? 0} ms    {tool.LastProvider}";
            lblSummary.ForeColor = blobs.Count > 0 ? Color.Firebrick : Color.ForestGreen;
        }
    }
}
