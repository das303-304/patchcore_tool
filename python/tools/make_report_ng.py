"""NG 检出评估报告（人工绘制缺陷）。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testimgs", "report_ng_eval.html")

# (文件, 改动强度maxdiff, >80像素数, OK分, NG分, 判定, 峰值距离, 召回)
ROWS = [
    ("107_20261008_101740.jpg", 163, 862, 0.407, 5.383, True, 1.4, 80.0),
    ("147_20261008_101941.jpg", 180, 2549, 0.560, 7.389, True, 4.0, 20.1),
    ("148_20261008_101942.jpg", 95, 64, 0.666, 4.587, True, 1.0, 46.4),
    ("179_20261008_102103.jpg", 103, 458, 0.603, 3.560, True, 1.0, 78.3),
    ("43_20261008_101441.jpg", 158, 670, 0.408, 5.096, True, 2.0, 96.2),
    ("89_20261008_101648.jpg", 110, 476, 0.399, 4.611, True, 0.0, 83.3),
    ("10_20261008_101306.jpg", 42, 0, 0.426, 1.209, True, 7.3, 33.3),
    ("49_20261008_101504.jpg", 44, 0, 0.401, 1.521, True, 1.4, 40.0),
    ("75_20261008_101609.jpg", 75, 0, 0.432, 4.031, True, 1.4, 58.6),
    ("72_20261008_101606.jpg", 81, 1, 0.423, 4.176, True, 1.0, 46.8),
    ("110_20261008_101740.jpg", 47, 0, 0.441, 2.163, True, 5.1, 36.6),
    ("116_20261008_101758.jpg", 79, 0, 0.407, 1.003, True, 12.2, 10.0),
    ("16_20261008_101331.jpg", 47, 0, 0.390, 0.507, False, 4.5, 28.6),
    ("134_20261008_101852.jpg", 59, 0, 0.379, 0.595, False, 2.0, 44.1),
]

# 被误放进 NG 目录的未修改文件
BOGUS = ("29_20261008_101406.jpg", 0.415, 0.415)


def rows_html():
    h = ["<table><thead><tr><th>NG 文件</th><th>改动强度<br>(灰阶)</th>"
         "<th>强改动<br>像素数</th><th>OK 原图分</th><th>NG 分</th>"
         "<th>判定</th><th>峰值距缺陷<br>(px)</th><th>热区召回</th></tr></thead><tbody>"]
    for f, md, n80, so, sn, hit, dist, rec in ROWS:
        cls = "" if hit else ' style="background:#ffebee"'
        v = '<span class="g">NG ✓</span>' if hit else '<span class="r">漏报</span>'
        h.append(f'<tr{cls}><td class="mono">{f}</td><td>{md}</td><td>{n80}</td>'
                 f'<td>{so:.3f}</td><td><b>{sn:.3f}</b></td><td>{v}</td>'
                 f'<td>{dist:.1f}</td><td>{rec:.1f}%</td></tr>')
    h.append("</tbody></table>")
    return "".join(h)


doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>NG 检出评估报告</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
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
  th,td {{ border:1px solid #e3e6ea; padding:7px 9px; text-align:center; }}
  th {{ background:#f0f2f5; font-weight:600; font-size:12px; }}
  td:first-child,th:first-child {{ text-align:left; }}
  .mono {{ font-family:Consolas,monospace; font-size:11.5px; }}
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
  .big {{ font-size:26px; font-weight:700; color:#1a56c4; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }}
  .stat {{ background:#f7f8fa; border:1px solid #e3e6ea; border-radius:8px;
    padding:14px; text-align:center; }}
</style></head><body><div class="wrap">

<h1>NG 检出评估报告</h1>
<div class="sub">14 张人工绘制缺陷 · 用生产记忆库检测 · <b>结论是乐观上界，不等于真实产线表现</b></div>

<div class="verdict">
<span class="k">结论：管线完全打通,检出 12/14 = 86%,OK 图零误报。</span><br>
但必须先说清楚:<b>这批 NG 是你在 OK 图上手动画的线</b>,
对比度最高达 180 个灰阶(纯黑线画在 ~130 灰底的料面上),
而真实划痕通常对比度低得多。<b>所以 86% 是"上限",不是"实际能力"。</b>
真正有价值的是下面第 4 节——<b>它验证了阈值 0.91/0.88 的位置是合理的</b>。
</div>

<div class="grid" style="margin-bottom:20px">
  <div class="stat"><div class="big">86%</div><div class="sm">检出率 12/14</div></div>
  <div class="stat"><div class="big" style="color:#2e7d32">0%</div><div class="sm">OK 图误报</div></div>
  <div class="stat"><div class="big" style="color:#ef6c00">6/6</div><div class="sm">强改动全检出</div></div>
</div>

<div class="card">
<h2>一、检测结果明细</h2>
{rows_html()}
<p class="sm">「改动强度」= OK 与 NG 的最大像素差（灰阶）。「强改动像素数」= 差值 &gt;80 的像素个数，
代表你画的那条线的"硬度"。<br>
「峰值距缺陷」= 热图峰值到最近缺陷像素的距离（掩码坐标系 224×224）。
「热区召回」= 热图前 1% 区域覆盖了多少缺陷像素。</p>

<div class="bad" style="padding:14px 18px;border-radius:8px;margin:14px 0">
<span class="k">发现一个数据问题:</span> <code>29_20261008_101406.jpg</code>
与它的 OK 原图<b>逐像素完全一致（maxdiff = 0）</b>,根本没被修改过。
它得 0.415 分（和 OK 一样）、峰值距缺陷 173 px,不是漏报,是<b>文件放错了</b>。
已从统计中排除。
</div>
</div>

<div class="card">
<h2>二、清晰的规律:改动越强,检出越稳</h2>
<h3>强改动（最大差值 ≥ 90 灰阶）—— 6/6 全部检出</h3>
<table><thead><tr><th>文件</th><th>改动强度</th><th>NG 分</th><th>峰值距缺陷</th><th>热区召回</th></tr></thead><tbody>
<tr><td class="mono">147_…</td><td>180</td><td>7.389</td><td>4.0 px</td><td>20.1%</td></tr>
<tr><td class="mono">107_…</td><td>163</td><td>5.383</td><td>1.4 px</td><td>80.0%</td></tr>
<tr><td class="mono">43_…</td><td>158</td><td>5.096</td><td>2.0 px</td><td>96.2%</td></tr>
<tr><td class="mono">89_…</td><td>110</td><td>4.611</td><td>0.0 px</td><td>83.3%</td></tr>
<tr><td class="mono">179_…</td><td>103</td><td>3.560</td><td>1.0 px</td><td>78.3%</td></tr>
<tr><td class="mono">148_…</td><td>95</td><td>4.587</td><td>1.0 px</td><td>46.4%</td></tr>
</tbody></table>
<p><b>6 张全部检出,分数 3.56~7.39,峰值距缺陷中位数仅 1.4 像素</b>
（224×224 掩码坐标系,相当于原图约 8 像素）—— 定位非常准。
其中 <code>43_</code> 的热区召回达 <b>96.2%</b>,几乎把整条线都圈住了。</p>

<h3>弱改动（最大差值 &lt; 90 灰阶）—— 8 张检出 6 张</h3>
<p>漏掉的两张:</p>
<ul>
<li><code>16_20261008_101331.jpg</code>　改动强度仅 47 灰阶,得分 <b>0.507</b></li>
<li><code>134_20261008_101852.jpg</code>　改动强度 59 灰阶,得分 <b>0.595</b></li>
</ul>
<p><b>两张漏报的分数落在了 OK 图正常分布区间内</b>（0.379~0.666）,
说明它们被检出的改动量已经小到与正常波动无法区分。</p>
</div>

<div class="card info">
<h2>三、最有价值的发现:阈值 0.91/0.88 位置合理</h2>
<p>这是本次评估真正的价值 —— 把两侧的分布放在一起看:</p>
<table><thead><tr><th>类别</th><th>分数范围</th><th>说明</th></tr></thead><tbody>
<tr><td>OK 图(库内自评)</td><td>0.379 ~ 0.666</td><td>偏低(图在库里)</td></tr>
<tr><td><b>OK 图(无偏)</b></td><td><b>0.766 ~ 0.773</b></td><td>交叉验证,真实正常上界</td></tr>
<tr><td style="background:#fff3cd"><b>阈值</b></td><td style="background:#fff3cd"><b>0.88 / 0.91</b></td><td style="background:#fff3cd">夹在中间</td></tr>
<tr><td>检出的 NG(最弱一张)</td><td>1.003</td><td>最弱的可检出改动</td></tr>
<tr><td>检出的 NG(最强)</td><td>7.389</td><td>—</td></tr>
</tbody></table>
<p><span class="k">正常上界 0.773 → 阈值 0.91 → 最弱可检出 1.003。</span>
阈值正好落在两者之间,<b>上不误报、下不放过中等以上改动</b>。
这说明之前用 mean+6σ 定的阈值经得起检验,不是拍脑袋来的。</p>
<p class="hl" style="padding:10px 14px;border-radius:8px">
留出的余量:距离正常上界 0.14,距离最弱可检出 0.09。
<span class="k">余量偏薄</span>——两边各只有 0.1 左右。想更稳的话可以把阈值降到 0.85,
代价是可能引入少量误报,需要你权衡。
</p>
</div>

<div class="card">
<h2>四、我排查过的两个"假问题"</h2>

<h3>假问题 1:是不是 JPEG 重压缩造成的分数升高?</h3>
<p>我算出 NG 图有 15~31% 的像素被改过,但平均只改 1.1 灰阶 —— 这看着像重压缩噪声。
如果分数升高是因为重压缩而不是画线,那整个评估就不成立。于是做了对照:</p>
<p><b>把 OK 图重新存一次 JPEG(不画任何东西),再评分:</b></p>
<pre>原始 OK 分   重压缩后   变化
0.407  →   0.423    +0.016
0.426  →   0.433    +0.008
0.560  →   0.555    -0.004
0.666  →   0.650    -0.016</pre>
<p><b>重压缩只影响 ±0.017,而画线让分数从 0.407 跳到 5.383。</b>
所以分数升高确实来自画的缺陷,<span class="k">评估成立</span>。那 15~31% 的微小差异是无害的 JPEG 噪声。</p>

<h3>假问题 2:热图峰值"偏离"缺陷</h3>
<p>我第一版指标算出来"峰值命中缺陷 0/11",看着像严重问题。
查下来是<b>我自己的指标错了</b>:你画的线在原图只有约 10 像素高,
缩到 224×224 掩码坐标系后只剩 <b>1.8 像素</b>,而掩码还经过 σ=4 的高斯平滑 ——
要求峰值精确命中这么细的线不合理。</p>
<p>换成"峰值到最近缺陷像素的距离"后,中位数只有 <b>1.4 像素</b>,
说明定位其实是准的。同时我还发现自己可视化里的坐标映射漏了中心裁剪的偏移,
一并修掉了。<span class="k">这两个都是我的 bug,不是算法的问题。</span></p>
</div>

<div class="card warn">
<h2>五、这批数据的局限（必须说清）</h2>
<ol>
<li><b class="r">画线不等于真实缺陷。</b>
你画的是高对比度的细线(最硬的一条改动了 180 个灰阶),
真实皮料缺陷多是压痕、擦伤、色差、毛孔异常 ——
形态更弥散、对比度更低。<b>所以 86% 是乐观上界。</b></li>

<li><b>没有缺陷形态的多样性。</b>14 张里基本都是"一条线",缺少面状缺陷、
色差类缺陷、边缘缺陷。而这些恰恰是 PatchCore 更难处理的类型。</li>

<li><b>无法算标准 AUROC。</b>因为 OK 图和 NG 图不是同一批(只有 14 对),
样本量不足以做统计意义上的指标。现在给的是"检出率"这种原始计数。</li>

<li><b>阈值余量偏薄</b>(正常上界 0.773 / 阈值 0.91 / 最弱可检出 1.003)。
如果真实缺陷正好落在 0.8~1.0 这个区间,会漏。而这个区间恰恰是最常见的
"轻微缺陷"所在。</li>
</ol>
</div>

<div class="card">
<h2>六、建议下一步</h2>
<ol>
<li><b>用真实缺陷再测一次。</b>哪怕只有 3~5 张真实缺陷料,
价值都超过这 14 张画线图。真实缺陷的分数分布才是定阈值的依据。</li>
<li><b>如果暂时拿不到真实缺陷</b>,可以画更多类型的:
面状污渍(大面积涂抹)、低对比度渐变(模拟色差)、边缘缺陷。
特别是<b>低对比度的</b>,用来找阈值下界。</li>
<li><b>阈值先维持 0.91/0.88</b>,按影子模式上线跑一段时间,
记录实际 NG 率与人工复核结果,用现场数据校准。</li>
<li>现场如果误报偏多,把阈值提到 1.0~1.1;如果漏报偏多,降到 0.80~0.85。
<b>但每次调整都要用同一批样本复测,不能凭感觉调。</b></li>
</ol>
</div>

<div class="card">
<h2>附:命令</h2>
<pre># NG 评估（含数据校验、阈值分析、可视化）
python eval_ng2.py

# 生成 4 栏对照可视化（OK原图 / NG / 热图 / 叠加带缺陷轮廓）
python vis_ng.py</pre>
<p class="sm">产物:<code>testimgs/report_ng_eval.html</code>(本报告)、
<code>testimgs/ng_vis/*.png</code>(6 组对照图)、
<code>testimgs/ng_eval.json</code>(明细数据)。</p>
</div>

</div></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written: {OUT} ({len(doc)} chars)")
