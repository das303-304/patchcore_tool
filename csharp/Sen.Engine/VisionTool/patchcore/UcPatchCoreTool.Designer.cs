namespace Sen.Engine.VisionTool
{
    partial class UcPatchCoreTool
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        /// <summary>
        /// 设计器支持所需的方法 —— 布局与 UcYolov5Tool.Designer.cs 保持同一套坐标体系：
        ///   左列：Label x≈12~27，控件 x=75，宽 74（限位类 62）
        ///   右列：Label x≈167~202，控件 x=230/287，宽 62~74
        ///   行距 30px，TabControl 页签 ItemSize 100×40，字体宋体 9pt
        /// 这样两个工具的界面观感完全一致，操作员不用重新适应。
        ///
        /// 与 YOLO 的差异只有一处：右侧显示区用 PictureBox，不用 HWndUnit（Halcon 控件）。
        /// </summary>
        private void InitializeComponent()
        {
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.tabControl = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.label1 = new System.Windows.Forms.Label();
            this.cbxImageSource = new System.Windows.Forms.ComboBox();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.labelModel = new System.Windows.Forms.Label();
            this.txtModel = new System.Windows.Forms.TextBox();
            this.btnModel = new System.Windows.Forms.Button();
            this.labelRoi = new System.Windows.Forms.Label();
            this.numRoi0 = new System.Windows.Forms.NumericUpDown();
            this.numRoi1 = new System.Windows.Forms.NumericUpDown();
            this.numRoi2 = new System.Windows.Forms.NumericUpDown();
            this.numRoi3 = new System.Windows.Forms.NumericUpDown();
            this.label7 = new System.Windows.Forms.Label();
            this.numThreshold = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.numNormalMean = new System.Windows.Forms.NumericUpDown();
            this.label3 = new System.Windows.Forms.Label();
            this.numBlobRatio = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.numMaxBlobs = new System.Windows.Forms.NumericUpDown();
            this.label5 = new System.Windows.Forms.Label();
            this.numMinArea = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.label17 = new System.Windows.Forms.Label();
            this.numMaxArea = new System.Windows.Forms.NumericUpDown();
            this.chxOKPicSave = new System.Windows.Forms.CheckBox();
            this.chxNGPicSave = new System.Windows.Forms.CheckBox();
            this.label8 = new System.Windows.Forms.Label();
            this.numCamChn = new System.Windows.Forms.NumericUpDown();
            this.label9 = new System.Windows.Forms.Label();
            this.numCutToCamDist = new System.Windows.Forms.NumericUpDown();
            this.label11 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.numCoef = new System.Windows.Forms.NumericUpDown();
            this.label12 = new System.Windows.Forms.Label();
            this.numXLeft = new System.Windows.Forms.NumericUpDown();
            this.label13 = new System.Windows.Forms.Label();
            this.numXRight = new System.Windows.Forms.NumericUpDown();
            this.label16 = new System.Windows.Forms.Label();
            this.numYTop = new System.Windows.Forms.NumericUpDown();
            this.label15 = new System.Windows.Forms.Label();
            this.numYBottom = new System.Windows.Forms.NumericUpDown();
            this.label14 = new System.Windows.Forms.Label();
            this.txtPicName = new System.Windows.Forms.TextBox();
            this.chxConfidence = new System.Windows.Forms.CheckBox();
            this.chxConfByMean = new System.Windows.Forms.CheckBox();
            this.chxYExtent = new System.Windows.Forms.CheckBox();
            this.chxGPU = new System.Windows.Forms.CheckBox();
            this.lblModelState = new System.Windows.Forms.Label();
            this.btnImport = new System.Windows.Forms.Button();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.listView1 = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.lblSummary = new System.Windows.Forms.Label();
            this.hWndShow = new Sen.Engine.ImageControl.HWndUnit();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.tabControl.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi0)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numThreshold)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numNormalMean)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numBlobRatio)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMaxBlobs)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMinArea)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMaxArea)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numCamChn)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numCutToCamDist)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numCoef)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numXLeft)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numXRight)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numYTop)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numYBottom)).BeginInit();
            this.tabPage4.SuspendLayout();
            this.SuspendLayout();
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            this.splitContainer1.Panel1.Controls.Add(this.tabControl);
            this.splitContainer1.Panel2.Controls.Add(this.hWndShow);
            this.splitContainer1.Size = new System.Drawing.Size(900, 400);
            this.splitContainer1.SplitterDistance = 391;
            this.splitContainer1.TabIndex = 0;
            // 
            // tabControl
            // 
            this.tabControl.Controls.Add(this.tabPage1);
            this.tabControl.Controls.Add(this.tabPage2);
            this.tabControl.Controls.Add(this.tabPage4);
            this.tabControl.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.tabControl.ItemSize = new System.Drawing.Size(100, 40);
            this.tabControl.Location = new System.Drawing.Point(0, 0);
            this.tabControl.Margin = new System.Windows.Forms.Padding(2);
            this.tabControl.Name = "tabControl";
            this.tabControl.SelectedIndex = 0;
            this.tabControl.Size = new System.Drawing.Size(391, 400);
            this.tabControl.TabIndex = 1;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.cbxImageSource);
            this.tabPage1.Controls.Add(this.label1);
            this.tabPage1.Location = new System.Drawing.Point(4, 44);
            this.tabPage1.Margin = new System.Windows.Forms.Padding(2);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(2);
            this.tabPage1.Size = new System.Drawing.Size(383, 352);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "图像源";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 11);
            this.label1.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(77, 12);
            this.label1.TabIndex = 2;
            this.label1.Text = "输入图像选择";
            // 
            // cbxImageSource
            // 
            this.cbxImageSource.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbxImageSource.FormattingEnabled = true;
            this.cbxImageSource.Location = new System.Drawing.Point(12, 33);
            this.cbxImageSource.Margin = new System.Windows.Forms.Padding(2);
            this.cbxImageSource.Name = "cbxImageSource";
            this.cbxImageSource.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
            | System.Windows.Forms.AnchorStyles.Right)));
            this.cbxImageSource.Size = new System.Drawing.Size(321, 20);
            this.cbxImageSource.TabIndex = 1;
            // 
            // tabPage2
            // 
            this.tabPage2.AutoScroll = true;
            this.tabPage2.Controls.Add(this.btnImport);
            this.tabPage2.Controls.Add(this.lblModelState);
            this.tabPage2.Controls.Add(this.chxGPU);
            this.tabPage2.Controls.Add(this.chxYExtent);
            this.tabPage2.Controls.Add(this.chxConfByMean);
            this.tabPage2.Controls.Add(this.chxConfidence);
            this.tabPage2.Controls.Add(this.txtPicName);
            this.tabPage2.Controls.Add(this.label14);
            this.tabPage2.Controls.Add(this.numYBottom);
            this.tabPage2.Controls.Add(this.label15);
            this.tabPage2.Controls.Add(this.numYTop);
            this.tabPage2.Controls.Add(this.label16);
            this.tabPage2.Controls.Add(this.numXRight);
            this.tabPage2.Controls.Add(this.label13);
            this.tabPage2.Controls.Add(this.numXLeft);
            this.tabPage2.Controls.Add(this.label12);
            this.tabPage2.Controls.Add(this.numCoef);
            this.tabPage2.Controls.Add(this.label10);
            this.tabPage2.Controls.Add(this.label11);
            this.tabPage2.Controls.Add(this.numCutToCamDist);
            this.tabPage2.Controls.Add(this.label9);
            this.tabPage2.Controls.Add(this.numCamChn);
            this.tabPage2.Controls.Add(this.label8);
            this.tabPage2.Controls.Add(this.chxNGPicSave);
            this.tabPage2.Controls.Add(this.chxOKPicSave);
            this.tabPage2.Controls.Add(this.numMaxArea);
            this.tabPage2.Controls.Add(this.label6);
            this.tabPage2.Controls.Add(this.label17);
            this.tabPage2.Controls.Add(this.numMinArea);
            this.tabPage2.Controls.Add(this.label5);
            this.tabPage2.Controls.Add(this.numMaxBlobs);
            this.tabPage2.Controls.Add(this.label4);
            this.tabPage2.Controls.Add(this.numBlobRatio);
            this.tabPage2.Controls.Add(this.label3);
            this.tabPage2.Controls.Add(this.numNormalMean);
            this.tabPage2.Controls.Add(this.label2);
            this.tabPage2.Controls.Add(this.numThreshold);
            this.tabPage2.Controls.Add(this.label7);
            this.tabPage2.Controls.Add(this.numRoi3);
            this.tabPage2.Controls.Add(this.numRoi2);
            this.tabPage2.Controls.Add(this.numRoi1);
            this.tabPage2.Controls.Add(this.numRoi0);
            this.tabPage2.Controls.Add(this.labelRoi);
            this.tabPage2.Controls.Add(this.btnModel);
            this.tabPage2.Controls.Add(this.txtModel);
            this.tabPage2.Controls.Add(this.labelModel);
            this.tabPage2.Location = new System.Drawing.Point(4, 44);
            this.tabPage2.Margin = new System.Windows.Forms.Padding(2);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(2);
            this.tabPage2.Size = new System.Drawing.Size(383, 352);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "检测参数";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // labelModel
            // 
            this.labelModel.AutoSize = true;
            this.labelModel.Location = new System.Drawing.Point(12, 17);
            this.labelModel.Name = "labelModel";
            this.labelModel.Size = new System.Drawing.Size(53, 12);
            this.labelModel.TabIndex = 0;
            this.labelModel.Text = "模型文件";
            // 
            // txtModel
            // 
            this.txtModel.Location = new System.Drawing.Point(75, 13);
            this.txtModel.Name = "txtModel";
            this.txtModel.Size = new System.Drawing.Size(250, 21);
            this.txtModel.TabIndex = 1;
            // 
            // btnModel
            // 
            this.btnModel.Location = new System.Drawing.Point(329, 12);
            this.btnModel.Name = "btnModel";
            this.btnModel.Size = new System.Drawing.Size(45, 23);
            this.btnModel.TabIndex = 2;
            this.btnModel.Text = "浏览";
            this.btnModel.UseVisualStyleBackColor = true;
            // 
            // labelRoi
            // 
            this.labelRoi.AutoSize = true;
            this.labelRoi.Location = new System.Drawing.Point(12, 47);
            this.labelRoi.Name = "labelRoi";
            this.labelRoi.Size = new System.Drawing.Size(53, 12);
            this.labelRoi.TabIndex = 3;
            this.labelRoi.Text = "ROI";
            // 
            // numRoi0
            // 
            this.numRoi0.Location = new System.Drawing.Point(75, 43);
            this.numRoi0.Maximum = new decimal(new int[] { 10000, 0, 0, 0 });
            this.numRoi0.Name = "numRoi0";
            this.numRoi0.Size = new System.Drawing.Size(60, 21);
            this.numRoi0.TabIndex = 4;
            // 
            // numRoi1
            // 
            this.numRoi1.Location = new System.Drawing.Point(139, 43);
            this.numRoi1.Maximum = new decimal(new int[] { 10000, 0, 0, 0 });
            this.numRoi1.Name = "numRoi1";
            this.numRoi1.Size = new System.Drawing.Size(60, 21);
            this.numRoi1.TabIndex = 5;
            // 
            // numRoi2
            // 
            this.numRoi2.Location = new System.Drawing.Point(222, 43);
            this.numRoi2.Maximum = new decimal(new int[] { 10000, 0, 0, 0 });
            this.numRoi2.Name = "numRoi2";
            this.numRoi2.Size = new System.Drawing.Size(60, 21);
            this.numRoi2.TabIndex = 6;
            // 
            // numRoi3
            // 
            this.numRoi3.Location = new System.Drawing.Point(286, 43);
            this.numRoi3.Maximum = new decimal(new int[] { 10000, 0, 0, 0 });
            this.numRoi3.Name = "numRoi3";
            this.numRoi3.Size = new System.Drawing.Size(60, 21);
            this.numRoi3.TabIndex = 7;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(12, 77);
            this.label7.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(41, 12);
            this.label7.TabIndex = 8;
            this.label7.Text = "异常分";
            // 
            // numThreshold
            // 
            this.numThreshold.DecimalPlaces = 3;
            this.numThreshold.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numThreshold.Location = new System.Drawing.Point(75, 73);
            this.numThreshold.Margin = new System.Windows.Forms.Padding(2);
            this.numThreshold.Maximum = new decimal(new int[] { 1000, 0, 0, 0 });
            this.numThreshold.Name = "numThreshold";
            this.numThreshold.Size = new System.Drawing.Size(74, 21);
            this.numThreshold.TabIndex = 9;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(167, 77);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(53, 12);
            this.label2.TabIndex = 10;
            this.label2.Text = "正常均值";
            // 
            // numNormalMean
            // 
            this.numNormalMean.DecimalPlaces = 3;
            this.numNormalMean.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numNormalMean.Location = new System.Drawing.Point(230, 73);
            this.numNormalMean.Maximum = new decimal(new int[] { 1000, 0, 0, 0 });
            this.numNormalMean.Name = "numNormalMean";
            this.numNormalMean.Size = new System.Drawing.Size(74, 21);
            this.numNormalMean.TabIndex = 11;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("宋体", 10.5F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.label3.Location = new System.Drawing.Point(199, 47);
            this.label3.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(21, 14);
            this.label3.TabIndex = 12;
            this.label3.Text = "→";
            this.label3.Visible = false;
            // 
            // numBlobRatio
            // 
            this.numBlobRatio.DecimalPlaces = 2;
            this.numBlobRatio.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            this.numBlobRatio.Location = new System.Drawing.Point(75, 103);
            this.numBlobRatio.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
            this.numBlobRatio.Name = "numBlobRatio";
            this.numBlobRatio.Size = new System.Drawing.Size(74, 21);
            this.numBlobRatio.TabIndex = 13;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(12, 107);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(53, 12);
            this.label4.TabIndex = 14;
            this.label4.Text = "区域系数";
            // 
            // numMaxBlobs
            // 
            this.numMaxBlobs.Location = new System.Drawing.Point(230, 103);
            this.numMaxBlobs.Maximum = new decimal(new int[] { 200, 0, 0, 0 });
            this.numMaxBlobs.Name = "numMaxBlobs";
            this.numMaxBlobs.Size = new System.Drawing.Size(74, 21);
            this.numMaxBlobs.TabIndex = 15;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(167, 107);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(53, 12);
            this.label5.TabIndex = 16;
            this.label5.Text = "最多区域";
            // 
            // numMinArea
            // 
            this.numMinArea.Location = new System.Drawing.Point(75, 133);
            this.numMinArea.Maximum = new decimal(new int[] { 10000000, 0, 0, 0 });
            this.numMinArea.Name = "numMinArea";
            this.numMinArea.Size = new System.Drawing.Size(74, 21);
            this.numMinArea.TabIndex = 17;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(12, 137);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(53, 12);
            this.label6.TabIndex = 18;
            this.label6.Text = "面积下限";
            // 
            // label17  （面积上限：原来这个输入框没有标签，看不出是干什么的）
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(167, 137);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(53, 12);
            this.label17.TabIndex = 46;
            this.label17.Text = "面积上限";
            // 
            // numMaxArea
            // 
            this.numMaxArea.Location = new System.Drawing.Point(230, 133);
            this.numMaxArea.Maximum = new decimal(new int[] { 10000000, 0, 0, 0 });
            this.numMaxArea.Name = "numMaxArea";
            this.numMaxArea.Size = new System.Drawing.Size(74, 21);
            this.numMaxArea.TabIndex = 19;
            // 
            // chxOKPicSave
            // 
            this.chxOKPicSave.AutoSize = true;
            this.chxOKPicSave.Location = new System.Drawing.Point(12, 166);
            this.chxOKPicSave.Margin = new System.Windows.Forms.Padding(2);
            this.chxOKPicSave.Name = "chxOKPicSave";
            this.chxOKPicSave.Size = new System.Drawing.Size(108, 16);
            this.chxOKPicSave.TabIndex = 20;
            this.chxOKPicSave.Text = "无缺陷图片保存";
            this.chxOKPicSave.UseVisualStyleBackColor = true;
            // 
            // chxNGPicSave
            // 
            this.chxNGPicSave.AutoSize = true;
            this.chxNGPicSave.Location = new System.Drawing.Point(167, 166);
            this.chxNGPicSave.Margin = new System.Windows.Forms.Padding(2);
            this.chxNGPicSave.Name = "chxNGPicSave";
            this.chxNGPicSave.Size = new System.Drawing.Size(96, 16);
            this.chxNGPicSave.TabIndex = 21;
            this.chxNGPicSave.Text = "缺陷图片保存";
            this.chxNGPicSave.UseVisualStyleBackColor = true;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(12, 197);
            this.label8.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(53, 12);
            this.label8.TabIndex = 22;
            this.label8.Text = "相机通道";
            // 
            // numCamChn
            // 
            this.numCamChn.Location = new System.Drawing.Point(75, 193);
            this.numCamChn.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
            this.numCamChn.Name = "numCamChn";
            this.numCamChn.Size = new System.Drawing.Size(74, 21);
            this.numCamChn.TabIndex = 23;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(167, 197);
            this.label9.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(53, 12);
            this.label9.TabIndex = 24;
            this.label9.Text = "切料距离";
            // 
            // numCutToCamDist
            // 
            this.numCutToCamDist.DecimalPlaces = 1;
            this.numCutToCamDist.Increment = new decimal(new int[] { 1, 0, 0, 65536 });
            this.numCutToCamDist.Location = new System.Drawing.Point(230, 193);
            this.numCutToCamDist.Maximum = new decimal(new int[] { 1000, 0, 0, 0 });
            this.numCutToCamDist.Name = "numCutToCamDist";
            this.numCutToCamDist.Size = new System.Drawing.Size(74, 21);
            this.numCutToCamDist.TabIndex = 25;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(310, 197);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(17, 12);
            this.label11.TabIndex = 26;
            this.label11.Text = "mm";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(12, 227);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(53, 12);
            this.label10.TabIndex = 27;
            this.label10.Text = "像素系数";
            // 
            // numCoef
            // 
            this.numCoef.DecimalPlaces = 6;
            this.numCoef.Increment = new decimal(new int[] { 1, 0, 0, 393216 });
            this.numCoef.Location = new System.Drawing.Point(75, 223);
            this.numCoef.Maximum = new decimal(new int[] { 10, 0, 0, 0 });
            this.numCoef.Name = "numCoef";
            this.numCoef.Size = new System.Drawing.Size(74, 21);
            this.numCoef.TabIndex = 28;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(12, 257);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(47, 12);
            this.label12.TabIndex = 29;
            this.label12.Text = "X左限位";
            // 
            // numXLeft
            // 
            this.numXLeft.Location = new System.Drawing.Point(75, 253);
            this.numXLeft.Maximum = new decimal(new int[] { 100000, 0, 0, 0 });
            this.numXLeft.Minimum = new decimal(new int[] { 100000, 0, 0, -2147483648 });
            this.numXLeft.Name = "numXLeft";
            this.numXLeft.Size = new System.Drawing.Size(74, 21);
            this.numXLeft.TabIndex = 30;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(167, 257);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(47, 12);
            this.label13.TabIndex = 31;
            this.label13.Text = "X右限位";
            // 
            // numXRight
            // 
            this.numXRight.Location = new System.Drawing.Point(230, 253);
            this.numXRight.Maximum = new decimal(new int[] { 100000, 0, 0, 0 });
            this.numXRight.Minimum = new decimal(new int[] { 100000, 0, 0, -2147483648 });
            this.numXRight.Name = "numXRight";
            this.numXRight.Size = new System.Drawing.Size(74, 21);
            this.numXRight.TabIndex = 32;
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(12, 287);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(59, 12);
            this.label16.TabIndex = 33;
            this.label16.Text = "Y顶部限位";
            // 
            // numYTop
            // 
            this.numYTop.Location = new System.Drawing.Point(75, 283);
            this.numYTop.Maximum = new decimal(new int[] { 100000, 0, 0, 0 });
            this.numYTop.Minimum = new decimal(new int[] { 100000, 0, 0, -2147483648 });
            this.numYTop.Name = "numYTop";
            this.numYTop.Size = new System.Drawing.Size(74, 21);
            this.numYTop.TabIndex = 34;
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(167, 287);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(59, 12);
            this.label15.TabIndex = 35;
            this.label15.Text = "Y底部限位";
            // 
            // numYBottom
            // 
            this.numYBottom.Location = new System.Drawing.Point(230, 283);
            this.numYBottom.Maximum = new decimal(new int[] { 100000, 0, 0, 0 });
            this.numYBottom.Minimum = new decimal(new int[] { 100000, 0, 0, -2147483648 });
            this.numYBottom.Name = "numYBottom";
            this.numYBottom.Size = new System.Drawing.Size(74, 21);
            this.numYBottom.TabIndex = 36;
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(12, 317);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(53, 12);
            this.label14.TabIndex = 37;
            this.label14.Text = "保存名称";
            // 
            // txtPicName
            // 
            this.txtPicName.Location = new System.Drawing.Point(75, 313);
            this.txtPicName.Name = "txtPicName";
            this.txtPicName.Size = new System.Drawing.Size(150, 21);
            this.txtPicName.TabIndex = 38;
            // 
            // chxConfidence
            // 
            this.chxConfidence.AutoSize = true;
            this.chxConfidence.Location = new System.Drawing.Point(12, 341);
            this.chxConfidence.Name = "chxConfidence";
            this.chxConfidence.Size = new System.Drawing.Size(138, 16);
            this.chxConfidence.TabIndex = 39;
            this.chxConfidence.Text = "写PLC用置信度(越大越好)";
            this.chxConfidence.UseVisualStyleBackColor = true;
            // 
            // chxConfByMean
            // 
            this.chxConfByMean.AutoSize = true;
            this.chxConfByMean.Location = new System.Drawing.Point(167, 341);
            this.chxConfByMean.Name = "chxConfByMean";
            this.chxConfByMean.Size = new System.Drawing.Size(108, 16);
            this.chxConfByMean.TabIndex = 40;
            this.chxConfByMean.Text = "按正常均值归一";
            this.chxConfByMean.UseVisualStyleBackColor = true;
            // 
            // chxYExtent
            // 
            this.chxYExtent.AutoSize = true;
            this.chxYExtent.Location = new System.Drawing.Point(12, 371);
            this.chxYExtent.Name = "chxYExtent";
            this.chxYExtent.Size = new System.Drawing.Size(138, 16);
            this.chxYExtent.TabIndex = 41;
            this.chxYExtent.Text = "额外输出Y极值(基址+偏移)";
            this.chxYExtent.UseVisualStyleBackColor = true;
            // 
            // chxGPU
            // 
            this.chxGPU.AutoSize = true;
            this.chxGPU.Location = new System.Drawing.Point(167, 371);
            this.chxGPU.Name = "chxGPU";
            this.chxGPU.Size = new System.Drawing.Size(152, 16);
            this.chxGPU.TabIndex = 42;
            this.chxGPU.Text = "优先GPU(失败回退CPU)";
            this.chxGPU.UseVisualStyleBackColor = true;
            // 
            // lblModelState
            // 
            this.lblModelState.AutoSize = false;
            this.lblModelState.Location = new System.Drawing.Point(12, 404);
            this.lblModelState.Name = "lblModelState";
            this.lblModelState.Size = new System.Drawing.Size(362, 62);
            this.lblModelState.TabIndex = 43;
            this.lblModelState.Text = "";
            // 
            // btnImport
            // 
            this.btnImport.Location = new System.Drawing.Point(12, 476);
            this.btnImport.Name = "btnImport";
            this.btnImport.Size = new System.Drawing.Size(200, 30);
            this.btnImport.TabIndex = 44;
            this.btnImport.Text = "导入模型到此相机";
            this.btnImport.UseVisualStyleBackColor = true;
            // 
            // tabPage4
            // 
            // Dock 顺序依据 z-order：index 大的先占位。lblSummary 要贴底，所以后加。
            this.tabPage4.Controls.Add(this.listView1);
            this.tabPage4.Controls.Add(this.lblSummary);
            this.tabPage4.Location = new System.Drawing.Point(4, 44);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage4.Size = new System.Drawing.Size(383, 352);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "测试结果";
            this.tabPage4.UseVisualStyleBackColor = true;
            // 
            // listView1
            // 
            this.listView1.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4,
            this.columnHeader5,
            this.columnHeader6,
            this.columnHeader7});
            this.listView1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.listView1.FullRowSelect = true;
            this.listView1.GridLines = true;
            this.listView1.HideSelection = false;
            this.listView1.Location = new System.Drawing.Point(3, 3);
            this.listView1.Name = "listView1";
            this.listView1.Size = new System.Drawing.Size(377, 300);
            this.listView1.TabIndex = 0;
            this.listView1.UseCompatibleStateImageBehavior = false;
            this.listView1.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1 ... 7
            // 
            this.columnHeader1.Text = "序号";
            this.columnHeader1.Width = 46;
            this.columnHeader2.Text = "X";
            this.columnHeader2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader2.Width = 52;
            this.columnHeader3.Text = "Y";
            this.columnHeader3.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader3.Width = 52;
            this.columnHeader4.Text = "宽";
            this.columnHeader4.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader4.Width = 46;
            this.columnHeader5.Text = "高";
            this.columnHeader5.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader5.Width = 46;
            this.columnHeader6.Text = "面积";
            this.columnHeader6.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader6.Width = 62;
            this.columnHeader7.Text = "峰值";
            this.columnHeader7.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader7.Width = 56;
            // 
            // lblSummary
            // 
            this.lblSummary.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.lblSummary.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Bold);
            this.lblSummary.Location = new System.Drawing.Point(3, 303);
            this.lblSummary.Name = "lblSummary";
            this.lblSummary.Size = new System.Drawing.Size(377, 46);
            this.lblSummary.TabIndex = 1;
            this.lblSummary.Text = "尚未运行";
            this.lblSummary.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // hWndShow
            // 
            this.hWndShow.CameraMessage = null;
            this.hWndShow.Dock = System.Windows.Forms.DockStyle.Fill;
            this.hWndShow.Location = new System.Drawing.Point(0, 0);
            this.hWndShow.Margin = new System.Windows.Forms.Padding(5);
            this.hWndShow.Name = "hWndShow";
            this.hWndShow.Size = new System.Drawing.Size(505, 400);
            this.hWndShow.TabIndex = 0;
            // 
            // UcPatchCoreTool
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.splitContainer1);
            this.Font = new System.Drawing.Font("宋体", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.Name = "UcPatchCoreTool";
            this.Size = new System.Drawing.Size(900, 400);
            // 关键：Dock=Fill 的 splitContainer1 必须放到 z-order 最前（index 0），
            // 这样它在 Dock 计算中最后处理、只填剩余区域。
            // 否则它会盖住基类 UcToolBase 的 toolStrip2 / groupBox2 / statusStrip1。
            this.Controls.SetChildIndex(this.splitContainer1, 0);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.tabControl.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi0)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numRoi3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numThreshold)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numNormalMean)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numBlobRatio)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMaxBlobs)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMinArea)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numMaxArea)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numCamChn)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numCutToCamDist)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numCoef)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numXLeft)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numXRight)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numYTop)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numYBottom)).EndInit();
            this.tabPage4.ResumeLayout(false);
            this.ResumeLayout(false);
        }

        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.TabControl tabControl;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ComboBox cbxImageSource;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.Label labelModel;
        private System.Windows.Forms.TextBox txtModel;
        private System.Windows.Forms.Button btnModel;
        private System.Windows.Forms.Label labelRoi;
        private System.Windows.Forms.NumericUpDown numRoi0;
        private System.Windows.Forms.NumericUpDown numRoi1;
        private System.Windows.Forms.NumericUpDown numRoi2;
        private System.Windows.Forms.NumericUpDown numRoi3;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.NumericUpDown numThreshold;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.NumericUpDown numNormalMean;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.NumericUpDown numBlobRatio;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.NumericUpDown numMaxBlobs;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.NumericUpDown numMinArea;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.NumericUpDown numMaxArea;
        private System.Windows.Forms.CheckBox chxOKPicSave;
        private System.Windows.Forms.CheckBox chxNGPicSave;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.NumericUpDown numCamChn;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.NumericUpDown numCutToCamDist;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.NumericUpDown numCoef;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.NumericUpDown numXLeft;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.NumericUpDown numXRight;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.NumericUpDown numYTop;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.NumericUpDown numYBottom;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TextBox txtPicName;
        private System.Windows.Forms.CheckBox chxConfidence;
        private System.Windows.Forms.CheckBox chxConfByMean;
        private System.Windows.Forms.CheckBox chxYExtent;
        private System.Windows.Forms.CheckBox chxGPU;
        private System.Windows.Forms.Label lblModelState;
        private System.Windows.Forms.Button btnImport;
        private System.Windows.Forms.TabPage tabPage4;
        private System.Windows.Forms.ListView listView1;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.ColumnHeader columnHeader3;
        private System.Windows.Forms.ColumnHeader columnHeader4;
        private System.Windows.Forms.ColumnHeader columnHeader5;
        private System.Windows.Forms.ColumnHeader columnHeader6;
        private System.Windows.Forms.ColumnHeader columnHeader7;
        private System.Windows.Forms.Label lblSummary;
        private ImageControl.HWndUnit hWndShow;
    }
}
