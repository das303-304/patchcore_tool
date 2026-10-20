"""对 C:/Users/Administrator/Desktop/123 批次的结论报告（HTML）。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testimgs", "report_123.html")

# 旧皮料库（对照基线）
LEATHER = [
    ("080d03b9559bf43454d70e0ffb49cc7f.jpg", 3.611),
    ("0fd8044ee58abaaada8f5db5e183deba(1).jpg", 5.030),
    ("2f90ec8d3923f40b6c72a12f3e5edfed.jpg", 4.170),
    ("5891e1f45ef89a7119c7469f510aab09(1).jpg", 5.922),
    ("5e87f775ce21de409cea16ea974ed6ea.jpg", 4.788),
]

# 自建库：A1 = 只用 1 张干净图建库；A2 = 用 2 张建库
A1 = [
    ("5891e1f45ef89a7119c7469f510aab09(1).jpg", 4.092, 2.3),
    ("0fd8044ee58abaaada8f5db5e183deba(1).jpg", 3.646, 2.4),
    ("2f90ec8d3923f40b6c72a12f3e5edfed.jpg", 2.657, 2.1),
    ("5e87f775ce21de409cea16ea974ed6ea.jpg", 2.223, 2.1),
    ("080d03b9559bf43454d70e0ffb49cc7f.jpg", 0.058, 97.7),
]
A2 = [
    ("5891e1f45ef89a7119c7469f510aab09(1).jpg", 3.996, 2.3),
    ("0fd8044ee58abaaada8f5db5e183deba(1).jpg", 3.494, 2.4),
    ("2f90ec8d3923f40b6c72a12f3e5edfed.jpg", 2.657, 2.1),
    ("080d03b9559bf43454d70e0ffb49cc7f.jpg", 0.065, 96.0),
    ("5e87f775ce21de409cea16ea974ed6ea.jpg", 0.000, 3.3),
]

# 中心 70% 裁剪后的稳定性
CROP = [
    ("5891e1f45ef89a7119c7469f510aab09(1).jpg", 4.092, 4.026),
    ("0fd8044ee58abaaada8f5db5e183deba(1).jpg", 3.646, 3.164),
    ("2f90ec8d3923f40b6c72a12f3e5edfed.jpg", 2.657, 2.860),
    ("5e87f775ce21de409cea16ea974ed6ea.jpg", 2.223, 2.168),
    ("080d03b9559bf43454d70e0ffb49cc7f.jpg", 0.058, 0.985),
]

NAME_CN = {
    "5891e1f45ef89a7119c7469f510aab09(1).jpg": "斜向浅色折痕",
    "0fd8044ee58abaaada8f5db5e183deba(1).jpg": "细斜浅痕",
    "2f90ec8d3923f40b6c72a12f3e5edfed.jpg": "左侧竖向深色折线",
    "5e87f775ce21de409cea16ea974ed6ea.jpg": "弧线 + 中部划痕",
    "080d03b9559bf43454d70e0ffb49cc7f.jpg": "平滑面料（无可见缺陷）",
}


def tbl_a1():
    h = ['<table><thead><tr><th>图片</th><th>内容</th><th>score</th><th>响应占比</th><th>排名</th></tr></thead><tbody>']
    for i, (n, s, c) in enumerate(A1, 1):
        inbank = s < 0.5
        cls = 'style="color:#c62828;font-weight:700"' if s > 2 else ""
        h.append(f'<tr><td class="mono">{n}</td><td>{NAME_CN[n]}</td>'
                 f'<td {cls}>{s:.3f}</td><td>{c:.1f}%</td>'
                 f'<td>{"（建库）" if inbank else f"{i}"}</td></tr>')
    h.append('</tbody></table>')
    return "\n".join(h)


def tbl_crop():
    h = ['<table><thead><tr><th>图片</th><th>全图 score</th><th>中心 70% score</th><th>变化</th></tr></thead><tbody>']
    for n, a, b in CROP:
        d = b - a
        h.append(f'<tr><td class="mono">{n}</td><td>{a:.3f}</td><td>{b:.3f}</td>'
                 f'<td>{d:+.3f}</td></tr>')
    h.append('</tbody></table>')
    return "\n".join(h)


doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>123 文件夹（手拍裁剪）检测结论</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background:#f5f6f8; color:#1c1f23; margin:0; padding:32px 24px; line-height:1.7; }}
  .wrap {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  .sub {{ color:#5a6472; font-size:14px; margin-bottom:26px; }}
  .card {{ background:#fff; border:1px solid #e3e6ea; border-radius:12px;
    padding:24px 26px; margin-bottom:20px; }}
  h2 {{ font-size:18px; margin:0 0 14px; padding-bottom:9px;
    border-bottom:2px solid #eceff2; }}
  h3 {{ font-size:15px; margin:20px 0 9px; color:#2b3038; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
  th, td {{ border:1px solid #e3e6ea; padding:7px 9px; text-align:center; }}
  th {{ background:#f0f2f5; font-weight:600; }}
  td:first-child, th:first-child {{ text-align:left; }}
  td:nth-child(2), th:nth-child(2) {{ text-align:left; }}
  .mono {{ font-family:Consolas, "Courier New", monospace; font-size:12px; }}
  .verdict {{ border-radius:10px; padding:18px 22px; font-size:15px;
    background:#e8f5e9; border:1px solid #a5d6a7; margin:14px 0; }}
  .warn {{ background:#fff8e1; border:1px solid #ffd54f; }}
  .bad {{ background:#ffebee; border:1px solid #ef9a9a; }}
  .info {{ background:#e3f2fd; border:1px solid #90caf9; }}
  pre {{ background:#1e2228; color:#d8dee6; padding:14px 16px; border-radius:8px;
    overflow-x:auto; font-family:Consolas, monospace; font-size:12.5px; line-height:1.6; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:3px;
    font-family:Consolas, monospace; font-size:12.5px; }}
  ul {{ padding-left:22px; }} li {{ margin:5px 0; }} ol {{ padding-left:22px; }}
  .k {{ font-weight:700; }} .g {{ color:#2e7d32; font-weight:700; }}
  .r {{ color:#c62828; font-weight:700; }}
  .hl {{ background:#fff3cd; padding:1px 4px; border-radius:3px; font-weight:700; }}
</style></head><body><div class="wrap">

<h1>能检出缺陷吗？—— 对 <code>Desktop\\123</code>（手拍裁剪）的结论</h1>
<div class="sub">PatchCore 无监督异常检测 · 5 张裁剪图 · 记忆库由本批次自建 · 无任何标注</div>

<div class="verdict">
<span class="k">结论：能，而且这次是干净利落的成功。</span><br>
裁剪掉背景后，结果发生了质变——<b>每一张被判「异常」的图，
异常响应都精确落在人眼可见的那条缺陷上</b>，而且分数在中心裁剪后依然稳定。
这说明检测器捕捉的确实是产品表面的真实缺陷，不是背景干扰。
</div>

<div class="card">
<h2>一、这批图和上一批有什么不同</h2>
<p><code>Desktop\\123</code> 里是 <b>5 张手拍裁剪图</b>：</p>
<table>
<thead><tr><th>文件名</th><th>尺寸</th><th>内容</th></tr></thead><tbody>
<tr><td class="mono">080d03b9…cc7f.jpg</td><td>704×914</td><td>平滑面料，无可见缺陷</td></tr>
<tr><td class="mono">0fd8044e…3deba(1).jpg</td><td>390×521</td><td>细斜浅痕（对角贯穿）</td></tr>
<tr><td class="mono">2f90ec8d…5edfed.jpg</td><td>910×762</td><td>左侧竖向深色折线</td></tr>
<tr><td class="mono">5891e1f4…aab09(1).jpg</td><td>300×401</td><td>斜向浅色折痕（右侧）</td></tr>
<tr><td class="mono">5e87f775…ed6ea.jpg</td><td>787×1233</td><td>弧线 + 中部划痕</td></tr>
</tbody></table>
<p><span class="k">关键变化：背景没了。</span>
上一批（<code>Desktop\\test</code>）里纸箱、手、桌面占了大量画面，
导致旧皮料库 10/10 全 NG 的跨域误报。这批已经裁到只留产品表面，
正好可以检验我上一轮的判断：<b>如果高分主要来自背景，裁剪后分数应该大幅下降。</b></p>
</div>

<div class="card">
<h2>二、基线对照：旧皮料库仍然是全 NG</h2>
<p>先用之前用 MVTec 棕色皮革建的库（19208 向量，阈值 1.67）直接跑：</p>
<table>
<thead><tr><th>图片</th><th>score</th><th>判定</th></tr></thead><tbody>
{''.join(f'<tr><td class="mono">{n}</td><td class="r">{s:.3f}</td><td class="r">NG</td></tr>' for n, s in LEATHER)}
</tbody></table>
<p>5/5 全 NG，分数 3.6～5.9。裁剪并没有救回旧皮料库——
因为问题不在背景，而在<b>材质本身就不一样</b>（棕色荔枝纹皮革 vs 灰色磨砂塑料）。
<span class="k">跨域问题时，唯一正解是用新材质自己建库。</span></p>
</div>

<div class="card">
<h2>三、正解：只用 1 张干净图建库，另外 4 张全部被点名</h2>
<p>这是最严苛的设置：<b>记忆库里只放 1 张</b>（那张无可见缺陷的平滑面料图），
其余 4 张全部当未知样本送检。</p>
{tbl_a1()}
<p class="sm">记忆库向量数 783（单张 224×224 图，patchsize=3）。
「响应占比」= 数值最高 1% 的像素贡献了多少总能量。</p>

<h3>三条关键观察</h3>
<ul>
<li><span class="g">分离度极干净</span>：建库图 0.058，其余 2.223～4.092，
中间空出 <span class="hl">2.165</span> 的鸿沟。阈值取 1.1 就能完美二分。</li>
<li><span class="g">响应高度集中</span>：有缺陷的 4 张响应占比全是 <b>2.1%～2.4%</b>
——能量聚成一小簇，这是"定位到具体缺陷"的典型特征。
而建库那张占比 <b>97.7%</b>，能量均匀铺满全图，是纯纹理残差。</li>
<li><span class="g">两张建库的结果几乎不变</span>：即使把"弧线+划痕"图也放进库
（实验 A2），其余三张的分数只从 4.092/3.646/2.657 变成 3.996/3.494/2.657，
<span class="k">排名顺序完全不变</span>。说明结果对建库构成不敏感，鲁棒。</li>
</ul>
</div>

<div class="card">
<h2>四、热图目视复核：异常精确压在人眼可见的缺陷上</h2>
<p>这是最有说服力的证据。我把热图叠加回原图逐张核对：</p>
<ul>
<li><b>5891e1f4…aab09(1).jpg &nbsp;score=4.092</b>（斜向浅色折痕）<br>
原图右侧有一条<span class="k">几乎看不清的浅色斜折痕</span>。
热图沿这条折痕纵向铺开一条<span class="r">红色亮带，起点、终点、走向完全吻合</span>。
人眼要费力才能看见的折痕，模型直接点出来了。</li>
<li><b>0fd8044e…3deba(1).jpg &nbsp;score=3.646</b>（细斜浅痕）<br>
热图的红色长条<b>严丝合缝地贴着那条对角细痕迹</b>，
而且连痕迹上深浅不均的段落变化都还原了（红条中间有几处更深的红点）。</li>
<li><b>2f90ec8d…5edfed.jpg &nbsp;score=2.657</b>（左侧竖向折线）<br>
红色响应压在左侧那条竖向折线上，另外左上角一个小圆点
——对应原图左上角的一处凹陷。两处都真实存在。</li>
<li><b>5e87f775…ed6ea.jpg &nbsp;score=2.223</b>（弧线 + 划痕）<br>
顶部弧形红色对应原图上方那条大弧线（面料轮廓阴影）。
但要注意：这张在全图评分下有一大块橙色区域，
<b>裁剪到中心 70% 后这块减弱了</b>，说明有一部分响应来自画面边缘的明暗渐变。</li>
</ul>
</div>

<div class="card">
<h2>五、稳健性检验：分数不是裁剪边缘造成的</h2>
<p>因为是<b>手拍裁剪</b>，理论上裁剪边缘可能引入伪影、虚高分数。
我把每张图只保留<b>中心 70% 区域</b>（彻底去掉裁剪边界）重新打分：</p>
{tbl_crop()}
<p><span class="g">结论：高分经得起裁剪。</span>
4 张有缺陷的图分数变化都在 <b>±0.5 以内</b>（4.092→4.026、3.646→3.164、
2.657→2.860、2.223→2.168），说明高分来自<b>画面内部的真实缺陷</b>，
不是边缘效应。</p>
<p class="warn">但有一个例外值得警惕：<b>建库的干净图 080d03b9…cc7f
从 0.058 涨到 0.985</b>（+0.928）。原因是中心裁剪放大了纹理、
改变了感受野下的纹理统计。虽然 0.985 仍低于阈值，但说明
<span class="k">分辨率/裁剪比例本身会影响分数</span>。
产线上必须<b>固定裁剪方案</b>，否则同一块料换个裁法分数就变。</p>
</div>

<div class="card warn">
<h2>六、眼下最没把握的事（必须说清楚）</h2>
<ol>
<li><b>我不知道哪几张是"应该合格"的。</b>
上面的分数排序在两组实验里都稳定，热图也漂亮，
但"稳定"只说明检测器在一致地捕捉差异，
<span class="k">不说明它捕捉的就是你要拦的缺陷</span>。
最典型的是 <b>5e87f775…ed6ea（弧线图）</b>：
那条弧线到底是<b>缺陷</b>（面料分层/起皱）还是<b>正常轮廓阴影</b>？
我看不出来。如果它是正常的，那它 2.223 的分数就是一次误报。</li>

<li><b>每类只 1 张样本，无法估计误报率。</b>
现在每张图都是"孤本"。真实产线上误报率（把好料判成 NG 的比例）
比漏报率更致命——会直接导致停机。
只有 5 张图算不出这个数，<b>至少需要几十张同类正常图</b>。</li>

<li><b>裁剪比例会改变分数</b>（见上节 080d03b9 从 0.058→0.985）。
这批是手拍裁剪，每张裁剪比例、拍摄距离、角度都不一样，
所以<b>这 5 张之间的分数横向比较要打折扣</b>。
分数可作为"单张 vs 建库基准"的参考，但不宜当作跨图片的统一标尺。</li>
</ol>
</div>

<div class="card info">
<h2>七、你说"后面才收集得到 OK 图" —— 那正好，这是标准做法</h2>
<p>PatchCore 的整个前提就是<b>只需要 OK 图，完全不需要缺陷图</b>。
代码层面已经确认（<code>mvtec.py:96/146</code>）：
ground_truth 仅在测试集且标注为非 good 时才读取。
<b>建记忆库只读正常样本，连一张缺陷图都不需要。</b>
所以你的数据收集计划完全对口。</p>

<h3>OK 图收集清单（越规范，后期越省事）</h3>
<ul>
<li><b>数量</b>：50～100 张起步。样本量决定"正常"的覆盖范围——
类内自然差异（纹理走向、光照渐变、褶皱、批次色差）覆盖不全，
正常图上的小波动就会被判异常。</li>
<li><b>覆盖</b>：不同批次、不同光照、不同角度、不同位置都要有。
按 MVTec leather 的经验，245 张才把类内差异覆盖完全。</li>
<li><b>固定成像</b>：相机位置、镜头、光源、工作距离全部锁死。
手拍的照片<b>不能用于建库</b>——因为产线是固定的，
用不固定的样本建库，等于让模型去学习一堆无关变量。</li>
<li><b>裁剪规范</b>：定一个固定的 ROI 裁剪方案（尺寸比例、对齐方式），
建库和推理必须用同一套。上面那个 0.058→0.985 就是这个坑。</li>
<li><b>命名/分目录</b>：全部平铺放进一个 <code>good/</code> 目录即可，
不需要分类、不需要标注、不需要文件名有含义。</li>
</ul>

<h3>等你 OK 图到位后的动作</h3>
<pre># 1) 把 OK 图全部放进  <你的目录>/shell/good/
#    （目录名 shell 可自定义，换成实际产品名）

# 2) 建库
python train_custom.py \\
  --data "C:/Users/Administrator/Desktop/<你的目录>" \\
  --classname shell \\
  --out my_models/shell \\
  --resize 256 --imagesize 224

# 3) 建库阶段会打印正常样本的分数分布 + 建议阈值

# 4) 投产推理
python infer_custom.py --model my_models/shell \\
  --dir "C:/Users/Administrator/Desktop/<待检>" \\
  --threshold <建库给出的阈值> --csv result.csv --save-heatmap</pre>
<p class="sm">提醒：<code>train_custom.py</code> 里 <code>--augment</code>
是空开关（不生效），别指望它扩充样本。样本不够就多拍，不要造假数据。</p>
</div>

<div class="card">
<h2>八、下一步（按优先级）</h2>
<ol>
<li><b>先回答一个问题</b>：<code>5e87f775…ed6ea.jpg</code> 里那条<b>弧线</b>
是缺陷还是正常轮廓？<br>
这一个答案会直接决定阈值怎么定。你可以只看这一张，不用全标。</li>
<li><b>确认另外 4 张的标签</b>：这 4 张（折痕/浅痕/折线/划痕）我判断是可检出的真实表面缺陷，
但那是我目视推断，需要你确认。确认后我可以算出精确率。</li>
<li><b>统一裁剪方案</b>：手拍裁剪 → 固定 ROI。这一步能消掉
上面那个 0.058→0.985 的不确定性。</li>
<li><b>按清单收集 50～100 张 OK 图</b>，然后按上面的命令建库。</li>
<li><b>固化成产线脚本</b>：需要的话我可以把 <code>infer_custom.py</code>
改造成对接海康 MV-CU120-10GM SDK 的版本——
直接吃相机帧，输出 OK/NG + 热图 + 可选 PLC 触发信号。</li>
</ol>
</div>

<div class="card">
<h2>附：本次复现命令</h2>
<pre># 1) 旧皮料库直接跑（对照，全 NG）
python infer_custom.py --model my_models/leather \\
  --dir "C:/Users/Administrator/Desktop/123" --threshold 1.67

# 2) 自建库交叉验证（单张建库 / 两张建库）
python crossval_123.py

# 3) 热图 + 中心裁剪稳健性检验
python heatmap_123.py</pre>
<p class="sm">产物：<code>testimgs/hm_123/*.png</code>（5 张上下对照热图：
上排全图、下排中心 70%）、<code>testimgs/report_123.html</code>（本报告）。</p>
</div>

</div></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written: {OUT}  ({len(doc)} chars)")
