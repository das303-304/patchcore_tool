"""OK 数据集标注审查报告 —— 发现被误标为 OK 的真实缺陷。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testimgs", "report_label_audit.html")

# 交叉验证打分（无偏）后高于阈值的图
OVER = [
    ("皮料相机2", "348_20261008_102824.jpg", 2.87, 0.88, "白色细划痕(约1.5cm)", "真实缺陷"),
    ("皮料相机1", "224_20261008_102254.jpg", 0.973, 0.91, "深色小痕迹(短划痕)", "真实缺陷"),
]

# 分布对比：139 张 vs 372 张
CMP = [
    ("皮料相机1", "139 张（当前库）", "0.544", "0.061", "0.773", "—"),
    ("皮料相机1", "372 张（全量）", "0.520", "0.068", "0.97~1.05", "剔除 224 后：约 0.82"),
    ("皮料相机2", "139 张（当前库）", "0.522", "0.059", "0.766", "—"),
    ("皮料相机2", "372 张（全量）", "0.503", "0.134", "≈2.9", "剔除 348 后：0.795"),
]

# 复核候选（分数 0.65 以上，供人工过目）
CAND1 = ["224_20261008_102254.jpg", "317_20261008_102651.jpg", "10_20261008_101306.jpg",
         "283_20261008_102528.jpg", "208_20261008_102212.jpg", "332_20261008_102731.jpg",
         "223_20261008_102253.jpg", "141_20261008_101925.jpg"]
CAND2 = ["348_20261008_102824.jpg", "68_20261008_101552.jpg", "158_20261008_101958.jpg",
         "214_20261008_102211.jpg", "98_20261008_101709.jpg", "233_20261008_102256.jpg",
         "185_20261008_102102.jpg", "274_20261008_102448.jpg"]


def tbl_over():
    h = ['<table><thead><tr><th>相机</th><th>文件名</th><th>分数</th><th>阈值</th>'
         '<th>热图指向的内容</th><th>判定</th></tr></thead><tbody>']
    for cam, f, s, thr, what, verdict in OVER:
        h.append(f'<tr style="background:#ffebee"><td>{cam}</td>'
                 f'<td class="mono">{f}</td><td class="r"><b>{s:.3f}</b></td>'
                 f'<td>{thr}</td><td>{what}</td><td class="r">{verdict}</td></tr>')
    return "".join(h) + "</tbody></table>"


def tbl_cmp():
    h = ['<table><thead><tr><th>相机</th><th>批次</th><th>均值</th><th>std</th>'
         '<th>最大分</th><th>剔除离群后</th></tr></thead><tbody>']
    for cam, batch, mean, std, mx, note in CMP:
        hl = ' style="background:#fff8e1"' if "全量" in batch else ""
        h.append(f'<tr{hl}><td>{cam}</td><td>{batch}</td><td>{mean}</td>'
                 f'<td>{std}</td><td><b>{mx}</b></td><td>{note}</td></tr>')
    return "".join(h) + "</tbody></table>"


doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>OK 数据集标注审查 —— 发现真实缺陷</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    background:#f5f6f8; color:#1c1f23; margin:0; padding:32px 24px; line-height:1.7; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:25px; margin:0 0 6px; }}
  .sub {{ color:#5a6472; font-size:14px; margin-bottom:24px; }}
  .card {{ background:#fff; border:1px solid #e3e6ea; border-radius:12px;
    padding:24px 26px; margin-bottom:20px; }}
  h2 {{ font-size:18px; margin:0 0 14px; padding-bottom:9px;
    border-bottom:2px solid #eceff2; }}
  h3 {{ font-size:15px; margin:18px 0 9px; color:#2b3038; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
  th,td {{ border:1px solid #e3e6ea; padding:7px 9px; text-align:center; }}
  th {{ background:#f0f2f5; font-weight:600; }}
  td:first-child,th:first-child {{ text-align:left; }}
  .mono {{ font-family:Consolas,monospace; font-size:11.5px; }}
  .verdict {{ border-radius:10px; padding:18px 22px; font-size:15px;
    background:#ffebee; border:1px solid #ef9a9a; margin:14px 0; }}
  .ok {{ background:#e8f5e9; border-color:#a5d6a7; }}
  .info {{ background:#e3f2fd; border:1px solid #90caf9; border-radius:10px;
    padding:16px 20px; margin:14px 0; }}
  .k {{ font-weight:700; }} .r {{ color:#c62828; font-weight:700; }}
  .g {{ color:#2e7d32; font-weight:700; }}
  pre {{ background:#1e2228; color:#d8dee6; padding:14px 16px; border-radius:8px;
    overflow-x:auto; font-family:Consolas,monospace; font-size:12.5px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:3px;
    font-family:Consolas,monospace; font-size:12.5px; }}
  ul,ol {{ padding-left:22px; }} li {{ margin:6px 0; }}
  .fig {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0; }}
  .fig img {{ width:100%; border:1px solid #e3e6ea; border-radius:8px; }}
  .cap {{ font-size:12px; color:#5a6472; margin-top:6px; }}
</style></head><body><div class="wrap">

<h1>OK 数据集标注审查 —— 发现 2 张被误标为 OK 的真实缺陷</h1>
<div class="sub">用全量 372×2 张交叉验证扫描 · 每张图由「没见过它」的库评分（无偏）</div>

<div class="verdict">
<span class="k">核心发现：你的 OK 数据集里有真实缺陷。</span><br>
两张图分数远超阈值，展开后能看到<b>真实的表面缺陷</b>（不是画上去的）：
<ul style="margin:10px 0 0">
<li><b>皮料相机2 / 348_20261008_102824.jpg</b> —— 白色细划痕，约 1.5 cm，<b>得分 ≈2.8~2.9</b></li>
<li><b>皮料相机1 / 224_20261008_102254.jpg</b> —— 深色小痕迹（短划痕），<b>得分 0.973</b></li>
</ul>
这两张如果进了记忆库，会把真实缺陷教成"正常"，<b>直接损害检出能力</b>。
</div>

<div class="card">
<h2>一、扫描结果</h2>
{tbl_over()}
<p>全量扫描下，<b>两个相机各只有 1 张图超过阈值</b>——分界非常干净，
不像数据整体有问题，而是<b>个别标注错误</b>。</p>

<h3>证据一：相机2 的白色划痕（得分 2.904）</h3>
<div class="fig">
  <div><img src="review_top/皮料相机2_01_348_20261008_102824.jpg.png">
    <div class="cap">4 栏：原图（含背景）/ ROI 裁剪 / 热图 / 叠加。
    热图是一个<b>紧致红点</b>，精确压在那条细线上。</div></div>
  <div><img src="ng_inspect/cam2_348_defect_zoom.png">
    <div class="cap">该处放大 4 倍：清楚可见一条<b>白色细划痕</b>，
    最亮值 201，周围约 123。</div></div>
</div>

<h3>证据二：相机1 的深色小痕迹（得分 0.973）</h3>
<div class="fig">
  <div><img src="review_top/皮料相机1_01_224_20261008_102254.jpg.png">
    <div class="cap">热图红点落在料面左侧。</div></div>
  <div><img src="ng_inspect/cam1_224_zoom.png">
    <div class="cap">该处放大：可见一个<b>深色小痕迹</b>，
    虽然很小，但确实是表面缺陷。</div></div>
</div>
</div>

<div class="card">
<h2>二、这解决了另一个悬而未决的问题</h2>
<div class="info">
之前我用你<b>手动画的线</b>评估检出能力时说过："画线对比度太高，86% 是乐观上界，
需要真实缺陷才能验证。"<br><br>
现在这两张就是<b>真实缺陷</b>——它们被模型自动抓出来了，
而且热图定位精确、分数远高于阈值。
<span class="k">这是第一次在真实缺陷上验证成功。</span><br><br>
<span class="g">相机2 的白色划痕得分 ≈2.8~2.9，是阈值 0.88 的 3.3 倍</span>，
余量很足。这对整个方案的信心提升，比那 14 张画线图加起来都大。
</div>
</div>

<div class="card">
<h2>三、关于"要不要重建库"—— 数据对比</h2>
{tbl_cmp()}
<p><span class="k">关键观察：</span>全量 372 张的最大分看起来变差了
（相机2 从 0.766 涨到 ≈2.9），但<b>这完全是那一张缺陷图造成的</b>——
剔除后第 2 名只有 0.795，和 139 张时的 0.766 几乎一样。</p>
<p>也就是说：<b>数据量的增加并没有让分布变差，是标注错误让它看起来变差。</b>
新增的 233 张成像条件与原有完全一致（均值 97.6 vs 98.5，暗区 22.6% vs 22.3%）。</p>
</div>

<div class="info" style="margin-bottom:20px">
<b>关于分数的运行波动：</b>coreset 采样含随机投影，同一模型两次运行分数会有
<b>±0.1 左右</b>的波动（实测相机2 的 348 图为 2.84~2.90）。所以看分数要看量级，
不要在 0.05 级别上纠结；但<b>图的排序是稳定的</b>。
</div>

<div class="card">
<h2>四、回答"需要重新训练吗？"</h2>

<h3>先纠正一个概念：PatchCore 不"训练"</h3>
<p>整个流程里<b>没有任何训练</b>：</p>
<ul>
<li>骨干网络是<b>冻结的</b> ImageNet 预训练权重，从头到尾不更新</li>
<li><code>fit()</code> 只做一件事——把正常样本的特征向量存进 FAISS 索引</li>
<li>没有反向传播、没有 loss、没有梯度、没有 epoch</li>
</ul>
<p>所以准确说法是<b>"重新建库"（rebuild memory bank）</b>，
本质是把新数据跑一遍特征提取再建索引，几秒钟的事，不存在"训练多久"的问题。</p>

<h3>那要不要重建？分三层看</h3>

<p><b>① 技术上：不是必须的。</b><br>
当前用 139/143 张建的库工作正常：14 张 NG 检出 12 张、282 张 OK 零误报。</p>

<p><b>② 但有件更紧迫的事：先清理标注。</b><br>
你的 OK 集里至少有 2 张真实缺陷。
<b>如果直接拿全量 372 张重建，这两张缺陷会被当成"正常"学进去</b>，
反而会削弱检出能力——这比"要不要重建"重要得多。</p>

<p><b>③ 值得重建的收益：比预期小。</b><br>
我原本以为样本翻 2.7 倍会明显收紧分布。实测下来：
剔除那 2 张后，372 张的分布与 139 张<b>基本持平</b>
（相机1 max 0.824 vs 0.773，相机2 0.795 vs 0.766）。
说明 139 张已经覆盖得不错了。<br>
<span class="k">所以：重建的收益有限，但清理标注的收益是决定性的。</span></p>
</div>

<div class="card">
<h2>五、建议的执行顺序</h2>
<ol>
<li><b>复核这 2 张，确认是缺陷</b>（我的判断是真实缺陷，但需要你确认——
    毕竟你了解产品标准，可能是允许的轻微瑕疵）。</li>
<li><b>把它们从 OK 集移到 NG 集</b>。它们比画的线有价值得多，
    是真实缺陷验证样本。</li>
<li><b>顺带过一遍其他候选图</b>（分数 0.65 以上的，见附录）。
    重点是相机2 的 68/158，相机1 的 317/10。</li>
<li><b>清理完再重建库</b>（372 张 - 剔除的）。</li>
<li><b>重跑验证定阈值</b>，再用 <code>eval_ng.py</code> 复测检出能力。</li>
</ol>
<p class="k">第 1~3 步是当前瓶颈，也是只有你能做的部分——模型的分数已经帮你把范围缩到最小了。</p>
</div>

<div class="card">
<h2>附录：可供复核的候选图（分数 0.65 以上）</h2>
<div class="fig">
<div>
<h3>皮料相机1</h3>
<pre>{chr(10).join(f'  {i+1}. {n:<32}' for i, n in enumerate(CAND1))}</pre>
</div>
<div>
<h3>皮料相机2</h3>
<pre>{chr(10).join(f'  {i+1}. {n:<32}' for i, n in enumerate(CAND2))}</pre>
</div>
</div>
<p class="sm">这些图的大多数应该只是纹理波动（分数 0.6~0.8 很接近正常均值 0.52），
但既然只有十几张，人工过一眼成本很低。</p>
</div>

<div class="card">
<h2>复现命令</h2>
<pre># 全量交叉验证扫描（每张图由没见过它的库评分）
python validate_ok_batch.py

# 列出最可疑的图并生成复核热图
python review_top.py --topk 10</pre>
<p class="sm">产物：<code>testimgs/report_label_audit.html</code>（本报告）、
<code>testimgs/review_top/</code>（复核热图）、
<code>testimgs/ng_inspect/cam2_348_defect_zoom.png</code>、
<code>testimgs/ng_inspect/cam1_224_zoom.png</code>（缺陷放大图）。</p>
</div>

</div></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written: {OUT} ({len(doc)} chars)")
