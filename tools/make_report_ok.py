"""OK 图批次（2026-10-08）建库与验证报告。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testimgs", "report_ok_20261008.html")

ROI_INFO = [
    ("皮料相机1", "139", "1800×1300 灰度", "(540, 30, 1780, 1270)", "540~1780 / 30~1270",
     "444 ~ 515", "黑色背景在左，占 22.3%"),
    ("皮料相机2", "143", "1800×1300 灰度", "(15, 30, 1400, 1270)", "15~1400 / 30~1270",
     "1419 ~ 1520", "黑色背景在右，占 12.5%"),
]

CV = [
    ("皮料相机1", "0.544", "0.061", "0.645", "0.696", "0.773",
     "10,897", "42.6 MB", "0.91"),
    ("皮料相机2", "0.522", "0.059", "0.617", "0.684", "0.766",
     "11,211", "43.8 MB", "0.88"),
]

ROI_AB = [
    ("皮料相机1", "不裁 ROI", "2.141", "1.275", "0.636", "两张图分数冲进缺陷量级"),
    ("皮料相机1", "裁 ROI", "0.773", "—", "0.645", "尖峰消失"),
    ("皮料相机2", "不裁 ROI", "0.701", "0.691", "0.597", "轻微抬高"),
    ("皮料相机2", "裁 ROI", "0.766", "—", "0.617", "基本持平"),
]

CROSS = [
    ("同域（相机1库 → 相机1图）", "0.411", "0.461", "库内含被测图，分数偏低"),
    ("跨域（相机1库 → 相机2图）", "0.902", "1.048", "从未见过，公平评分"),
    ("同域（相机2库 → 相机2图）", "0.394", "0.446", "库内含被测图，分数偏低"),
    ("跨域（相机2库 → 相机1图）", "0.988", "1.216", "从未见过，公平评分"),
]

SPEED = [
    ("GPU (RTX 3060)", "134.8 ms", "7.42", "13.5%"),
    ("CPU (8核16线程)", "226.0 ms", "4.43", "22.6%"),
]


def table(headers, rows, mono_first=True):
    h = ["<table><thead><tr>"]
    for x in headers:
        h.append(f"<th>{x}</th>")
    h.append("</tr></thead><tbody>")
    for r in rows:
        h.append("<tr>")
        for i, c in enumerate(r):
            cls = ' class="mono"' if (mono_first and i == 0) else ""
            h.append(f"<td{cls}>{c}</td>")
        h.append("</tr>")
    return "".join(h) + "</tbody></table>"


doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>OK 图建库报告 2026-10-08</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background:#f5f6f8; color:#1c1f23; margin:0; padding:32px 24px; line-height:1.7; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
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
  .mono {{ font-family:Consolas,"Courier New",monospace; font-size:12px; }}
  .verdict {{ border-radius:10px; padding:18px 22px; font-size:15px;
    background:#e8f5e9; border:1px solid #a5d6a7; margin:14px 0; }}
  .warn {{ background:#fff8e1; border:1px solid #ffd54f; }}
  .bad {{ background:#ffebee; border:1px solid #ef9a9a; }}
  .info {{ background:#e3f2fd; border:1px solid #90caf9; }}
  pre {{ background:#1e2228; color:#d8dee6; padding:14px 16px; border-radius:8px;
    overflow-x:auto; font-family:Consolas,monospace; font-size:12.5px; line-height:1.6; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:3px;
    font-family:Consolas,monospace; font-size:12.5px; }}
  ul,ol {{ padding-left:22px; }} li {{ margin:5px 0; }}
  .k {{ font-weight:700; }} .g {{ color:#2e7d32; font-weight:700; }}
  .r {{ color:#c62828; font-weight:700; }}
  .hl {{ background:#fff3cd; padding:1px 4px; border-radius:3px; font-weight:700; }}
  .big {{ font-size:22px; font-weight:700; color:#1a56c4; }}
</style></head><body><div class="wrap">

<h1>OK 图建库报告 —— <code>2026-10-08</code> 批次</h1>
<div class="sub">282 张工业相机 OK 图 · 两个相机 · 无监督建库 · 无标注 · 零误报</div>

<div class="verdict">
<span class="k">结论：数据质量很好，两个记忆库已建好，本批次零误报。</span><br>
但要先说一件事：<b>这批图不能确认能检出缺陷</b>——因为它们全是 OK 图，
没有任何缺陷样本可验证。我能证明的是"正常料不会被误报"，
<b>检出能力这一半还是空白</b>。详见末尾"未验证事项"。
</div>

<div class="card">
<h2>一、数据概况</h2>
{table(["相机", "张数", "格式", "ROI", "ROI 范围(x/y)", "物料边缘漂移", "取景特点"], ROI_INFO)}
<p><span class="k">两个关键事实：</span></p>
<ul>
<li><b>灰度图（mode L）</b>，1800×1300，两个相机格式完全一致 —— 标准工业相机输出，
比之前的手拍图规整得多。</li>
<li><b>两个相机的取景是镜像的</b>：相机1 黑边在左（占 22.3%），相机2 黑边在右（占 12.5%）。
这意味着<b>它们不能共用一套 ROI，也不能共用一个记忆库</b>。</li>
<li><b>物料边缘会漂移</b>：相机1 亮区起点在 444~515 之间浮动（漂移 71px），
相机2 终点在 1419~1520（漂移 101px）。ROI 必须留足余量，否则会把黑边裁进来。</li>
</ul>
</div>

<div class="card">
<h2>二、ROI 裁剪是必须的 —— 有硬证据</h2>
<p>黑边/背景是最大误报源。我做了对照实验（双向交叉验证，分数无偏）：</p>
{table(["相机", "处理", "最大分", "次高", "p95", "说明"], ROI_AB)}
<p class="bad" style="padding:12px 16px; border-radius:8px; margin:12px 0">
<span class="k">注意第一行：</span>相机1 不裁 ROI 时，有两张图分数冲到 <b>2.141 和 1.275</b>。
而之前实测真实缺陷的量级是 <b>2~4</b> —— 也就是说这两张正常料会被判成缺陷。
裁掉 ROI 后尖峰完全消失（最大 0.773）。<br>
<span class="k">裁 ROI 不是"优化"，是"必需"。</span>
</p>
<p>根因：物料边缘位置漂移，固定的 ROI 会把边缘处的黑边像素裁进来，
产生一个强边缘 —— 模型看到"没见过的边缘"就报警。</p>
</div>

<div class="card">
<h2>三、两个相机能不能共用一个库？—— 不能</h2>
<p>这个问题直接对应你之前问的"每台机器环境光/打光/物距/ROI 都不同怎么办"。
我用你这批真实数据做了跨相机测试：</p>
{table(["评测方式", "均值", "最大", "说明"], CROSS)}
<p><span class="k">公平对比</span>（两边都是"模型没见过的图"）：</p>
<ul>
<li>同相机未见图：均值 ≈ <b>0.53</b>，最大 ≈ <b>0.78</b>（来自留出验证）</li>
<li>跨相机未见图：均值 ≈ <b>0.95</b>，最大 ≈ <b>1.22</b></li>
<li><span class="k">放大约 1.5～1.8 倍</span></li>
</ul>
<p>1.22 已经超过单机阈值 0.91。<b>如果把相机1 的库装到相机2 上，会产生成片误报。</b></p>
<p class="hl" style="padding:10px 14px; border-radius:8px">
结论：<b>每台机器必须独立建库、独立标定阈值。</b>
这就是我之前说的"第三层防线"，现在有你的实测数据支撑了。
</p>
</div>

<div class="card">
<h2>四、建好的记忆库</h2>
{table(["相机", "无偏均值", "std", "p95", "p99", "最大", "记忆库向量", "文件大小", "建议阈值"], CV)}
<p class="sm">分数来自 <b>2 折交叉验证</b>：A 组建库评 B 组、B 组建库评 A 组。
每张图都由"没见过它"的库评分，所以是无偏的。<br>
（不能拿"建库图自己的分数"当基准 —— 图自己的特征就在库里，分数被人为压低，
那样比必然得出虚高的误报率。上一轮我就差点犯这个错。）</p>

<h3>阈值建议</h3>
<ul>
<li>相机1：<b>mean+6σ = 0.911</b>　（保守可取 0.95）</li>
<li>相机2：<b>mean+6σ = 0.877</b>　（保守可取 0.90）</li>
</ul>
<p>按这两个阈值，<b>282 张 OK 图全部通过，零误报</b>。而且留有相当余量
（最大观测 0.773 / 0.766，阈值 0.91 / 0.88）。</p>

<h3>产物</h3>
<pre>models/cam1/   42.6 MB   10,897 向量  nnscorer_search_index.faiss + patchcore_params.pkl
models/cam2/   43.8 MB   11,211 向量  同上</pre>
</div>

<div class="card">
<h2>五、我复核了"最可疑的 8 张" —— 都不是缺陷</h2>
<p>我用无偏分数排了序，把每个相机最可疑的 8 张出了热图（4 栏：原图含背景 / ROI裁剪 / 热图 / 叠加）。</p>
<p><span class="k">结果：这些图人眼确认都是正常料面，没有可见缺陷。</span>
热图的亮斑集中在<b>四角</b>，而四角是光照梯度的极值处 —— 属于成像残差，不是产品缺陷。</p>

<h3>顺带查出一个光照问题</h3>
<p>我统计了 ROI 内部的亮度分布：</p>
<table><thead><tr><th>相机</th><th>顶部均值</th><th>底部均值</th><th>上下差</th><th>左边缘</th><th>中心</th></tr></thead><tbody>
<tr><td class="mono">皮料相机1</td><td>108.5</td><td>135.0</td><td class="r">+26.5</td><td>97.5</td><td>132.9</td></tr>
<tr><td class="mono">皮料相机2</td><td>98.1</td><td>122.6</td><td class="r">+24.3</td><td>112.3</td><td>119.5</td></tr>
</tbody></table>
<p><b>两个相机都有约 25 个灰阶的上下亮度差</b> —— 这是光源照度不均匀（不是相机问题）。</p>
<p>我试过用"收窄 ROI"来规避：左右方向的梯度能压下去（+27.4 → +5.1），
但<b>上下方向压不掉</b>（一直保持 +25），而且收窄后分数分布反而变差
（相机2 最大分 0.766 → 0.942）。<span class="k">所以不建议收窄 ROI，
应该从光源本身解决照度均匀性。</span></p>
</div>

<div class="card info">
<h2>六、一个好消息：光照波动不影响检测</h2>
<p>相机1 的整图亮度在批次内波动 17.7 个灰阶（min 90.8 / max 108.5），
我担心这会推高误报。于是算了相关性：</p>
<table><thead><tr><th>相机</th><th>score vs 料区亮度</th><th>score vs 料区对比度</th><th>score vs 顶底梯度</th></tr></thead><tbody>
<tr><td class="mono">皮料相机1</td><td>+0.085</td><td>+0.028</td><td>+0.048</td></tr>
<tr><td class="mono">皮料相机2</td><td>+0.100</td><td>−0.067</td><td>+0.055</td></tr>
</tbody></table>
<p><b>三个相关系数全部接近 0（|r| &lt; 0.11）。</b>
说明分数与光照亮度、对比度、梯度都<b>不相关</b> ——
模型对这批数据的光照波动是稳健的，光照漂移不是高分来源。
这比我预期的结果要好。</p>
</div>

<div class="card">
<h2>七、速度：CPU 也够用</h2>
<p>用生产模型实测（记忆库约 11000 向量，ROI 裁剪后 1240×1240）：</p>
{table(["设备", "端到端耗时", "FPS", "占 1s 节拍"], SPEED)}
<p>产线节拍 1 秒，<b>CPU 只占 22.6%，GPU 占 13.5%</b>，都有充足余量。
<span class="k">工控机不需要配独显。</span></p>
</div>

<div class="card warn">
<h2>八、未验证事项（必须说清楚）</h2>
<ol>
<li><b class="r">检出能力完全未验证。</b>
这 282 张全是 OK 图。我能证明"正常料不误报"，
但<b>无法证明"有缺陷时能报警"</b>。
阈值 0.91 是基于正常分布定的，<b>它是否落在缺陷分之上，没有数据支撑</b>。
这是当前最大的空白。</li>

<li><b>阈值可能偏保守。</b>mean+6σ 是"宁可漏报不可误报"的取向。
如果实际缺陷分数在 1.0~1.5 之间，0.91 勉强够；
如果缺陷很轻微（比如浅划痕只有 0.8），就会漏掉。</li>

<li><b>光照均匀性待改善。</b>约 25 灰阶的上下差虽然目前没造成误报
（因为所有图都有这个梯度，模型学会了），但它吃掉了分数余量。
如果哪天灯管老化、换批次光源，梯度变了，误报可能突然出现。</li>

<li><b>跨相机比例只测了两个相机。</b>1.5~1.8 倍这个数是从两台机得来的，
第三台机可能更好也可能更差。</li>
</ol>
</div>

<div class="card">
<h2>九、建议的下一步</h2>
<ol>
<li><b>最关键：找一些确认有缺陷的料，拍下来测一次。</b>
哪怕只有 5~10 张、哪怕只是人工制造（划一道、捏一下）。
有了缺陷分数，才能验证 0.91 这个阈值是否合理，以及确定漏报边界。
<span class="k">现在整个方案卡在这一步。</span></li>

<li><b>其余机器上线时，照相机1/2 的做法独立建库</b>：
每台机收 50~100 张 OK 图 → 定 ROI → 建库 → 用留出法验证 → 定阈值。
流程脚本都已就绪。</li>

<li><b>改善光源照度均匀性</b>（可选但推荐）。
目标是把上下差从 25 灰阶压到 10 以内。
这会让正常分数分布更紧，阈值余量更大，抗老化能力更强。</li>

<li>把当前阈值写进产线程序，先 <b>影子模式</b>跑一段时间
（只记录判定、不触发停机），观察实际 NG 率是否符合预期。</li>
</ol>
</div>

<div class="card">
<h2>附：完整命令</h2>
<pre># 建库（每台机一条，ROI 必须按实际相机填）
python train_custom.py --data ok_data --classname 皮料相机1 \\
  --out models/cam1 --roi 540,30,1780,1270 --resize 256 --imagesize 224

# 留出法验证（判断误报率、给阈值建议）
python validate_ok_batch.py

# 跨相机验证（判断库能否通用）
python crosscam_validate.py

# 交叉验证打分 + 最可疑图复核热图
python review_top.py --topk 8

# 生产推理
python infer_custom.py --model models/cam1 \\
  --dir &lt;待检目录&gt; --roi 540,30,1780,1270 --threshold 0.91 \\
  --csv result.csv --save-heatmap</pre>
<p class="sm">报告与热图：<code>testimgs/report_ok_20261008.html</code>、
<code>testimgs/review_top/</code>（16 张复核热图）、
<code>testimgs/ok_cam1_scores.csv</code>、<code>testimgs/ok_cam2_scores.csv</code>。</p>
</div>

</div></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written: {OUT} ({len(doc)} chars)")
