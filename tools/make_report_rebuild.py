"""标注修正后的重建与阈值重定 —— 交付报告。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testimgs", "report_rebuild.html")

# 标注修正
FIX = [
    ("皮料相机2", "348_20261008_102824.jpg", "true_defect",
     "白色细划痕(约1.5cm)", "≈2.8~2.9", "你已删除 —— 确认是标注失误"),
    ("皮料相机1", "224_20261008_102254.jpg", "ok",
     "深色纤维/天然痕迹", "1.025", "你确认为 OK —— 属可接受的自然变异"),
]

# 模型对比
MODELS = [
    ("皮料相机1", "139 张", "10,897", "42.6 MB", "0.544", "0.061", "0.773", "0.91"),
    ("皮料相机1", "372 张（现用）", "29,164", "113.9 MB", "0.520", "0.068", "1.025", "1.05"),
    ("皮料相机2", "143 张", "11,211", "43.8 MB", "0.522", "0.059", "0.766", "0.88"),
    ("皮料相机2", "371 张（现用）", "29,086", "113.6 MB", "0.497", "0.057", "0.772", "0.84"),
]

# 阈值权衡
TRADE1 = [
    ("0.85", "1 张", "9/10", "1 张", "误报+漏报"),
    ("0.91（旧）", "1 张", "9/10", "1 张", "误报+漏报"),
    ("1.00", "1 张", "9/10", "1 张", "误报+漏报"),
    ("1.05（现用）", "0", "9/10", "1 张", "零误报"),
    ("1.10", "0", "8/10", "2 张", "零误报但灵敏度下降"),
    ("1.30", "0", "8/10", "2 张", "同上"),
]
TRADE2 = [
    ("0.70", "1 张", "3/4", "1 张", "误报+漏报"),
    ("0.78", "0", "3/4", "1 张", "零误报"),
    ("0.84（现用）", "0", "3/4", "1 张", "零误报"),
    ("0.90", "0", "3/4", "1 张", "零误报"),
    ("0.95", "0", "2/4", "2 张", "零误报但灵敏度下降"),
]


def rows(data, hi=None):
    h = []
    for r in data:
        hl = ' style="background:#e8f5e9"' if hi and hi in r[0] else ""
        h.append(f"<tr{hl}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return "".join(h)


doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>标注修正后重建与阈值重定</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    background:#f5f6f8; color:#1c1f23; margin:0; padding:32px 24px; line-height:1.7; }}
  .wrap {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:25px; margin:0 0 6px; }}
  .sub {{ color:#5a6472; font-size:14px; margin-bottom:24px; }}
  .card {{ background:#fff; border:1px solid #e3e6ea; border-radius:12px;
    padding:24px 26px; margin-bottom:20px; }}
  h2 {{ font-size:18px; margin:0 0 14px; padding-bottom:9px;
    border-bottom:2px solid #eceff2; }}
  h3 {{ font-size:15px; margin:18px 0 9px; color:#2b3038; }}
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
  .k {{ font-weight:700; }} .r {{ color:#c62828; font-weight:700; }}
  .g {{ color:#2e7d32; font-weight:700; }} .b {{ color:#1a56c4; font-weight:700; }}
  .hl {{ background:#fff3cd; padding:2px 6px; border-radius:4px; font-weight:700; }}
  pre {{ background:#1e2228; color:#d8dee6; padding:14px 16px; border-radius:8px;
    overflow-x:auto; font-family:Consolas,monospace; font-size:12.5px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:3px;
    font-family:Consolas,monospace; font-size:12.5px; }}
  ul,ol {{ padding-left:22px; }} li {{ margin:6px 0; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="wrap">

<h1>标注修正后的重建与阈值重定</h1>
<div class="sub">两个记忆库已用全量数据重建 · 阈值已重新标定 · 相机1 暴露一个需要决策的取舍</div>

<div class="verdict">
<span class="k">已完成：模型重建、阈值重定、NG 复测。</span><br>
检出能力保持 <b>12/15 = 80%</b>，定位精度还略有提升（66.7% → 73.3%），OK 图零误报。<br>
但相机1 出现一个<b>必须由你决策的取舍</b>：可接受的"深色纤维"自然变异与最弱缺陷的分数重叠了。
</div>

<div class="card">
<h2>一、标注修正结果</h2>
<table><thead><tr><th>相机</th><th>文件</th><th>内容</th><th>分数</th><th>处理</th></tr></thead><tbody>
<tr style="background:#ffebee"><td>皮料相机2</td><td class="mono">348_20261008_102824.jpg</td>
  <td>白色细划痕(约1.5cm)</td><td class="r">≈2.8~2.9</td><td>你已删除 —— 确认是失误</td></tr>
<tr><td>皮料相机1</td><td class="mono">224_20261008_102254.jpg</td>
  <td>深色纤维 / 天然痕迹</td><td>1.025</td><td>你确认为 OK —— 可接受的自然变异</td></tr>
</tbody></table>

<h3>一个重要验证：224 在库内时分数会掉下来</h3>
<p>我用全量 372 张（含 224）建库后重新测它：</p>
<pre>224_20261008_102254.jpg   在库内得分 0.3908   （原本 1.025）
全量在库内分布: 均值 0.3933  最大 0.4454</pre>
<p><b>这说明 1.025 是"不在库里"时的分数</b>。生产环境只要库里有这类样本，
它就不再报警 —— <span class="k">这也意味着：如果这类深色纤维在产线上经常出现，
把它们收进库就能自然解决，不用动阈值。</span></p>
</div>

<div class="card">
<h2>二、重建后的模型</h2>
<table><thead><tr><th>相机</th><th>样本量</th><th>记忆库向量</th><th>文件大小</th>
<th>正常均值</th><th>std</th><th>正常上界</th><th>阈值</th></tr></thead><tbody>
{rows(MODELS, hi="现用")}
</tbody></table>
<p class="sm">「正常上界」来自双向交叉验证的留出分数（无偏）。旧模型已备份为
<code>models/cam1_139bak</code> / <code>cam2_143bak</code>。</p>

<h3>关键变化：相机2 变干净了</h3>
<p>删除 348 后，相机2 的正常上界从 <b>2.867 降到 0.772</b>，
标准差从 0.134 降到 <b>0.057</b> —— 和之前 143 张时的 0.766 / 0.059 基本一致。<br>
<span class="k">证明之前那个"数据变多反而更差"的现象，完全是那一张误标缺陷造成的，与样本量无关。</span></p>
</div>

<div class="card warn">
<h2>三、相机1 的取舍（需要你决策）</h2>
<p>相机1 的情况比相机2 复杂。把两侧分布摊开看：</p>
<table><thead><tr><th>类别</th><th>分数</th><th>说明</th></tr></thead><tbody>
<tr><td>正常样本均值</td><td>0.520</td><td>372 张</td></tr>
<tr><td>正常样本 p99</td><td>0.695</td><td>—</td></tr>
<tr><td class="hl">正常上界（224 深色纤维）</td><td class="hl">1.025</td><td>你确认 OK</td></tr>
<tr><td class="hl">最弱的可检出 NG</td><td class="hl">1.088</td><td>10_ 那张</td></tr>
<tr><td>其余 NG</td><td>1.404 ~ 7.362</td><td>7 张</td></tr>
</tbody></table>
<p><span class="k r">分离窗口只有 1.088 − 1.025 = 0.063。</span>
而分数存在 <b>±0.1</b> 的运行波动（coreset 随机投影导致）——
<span class="k">窗口比噪声还窄，理论上无法可靠区分。</span></p>

<h3>阈值权衡（相机1）</h3>
<table><thead><tr><th>阈值</th><th>OK 误报</th><th>NG 检出</th><th>漏报</th><th>说明</th></tr></thead><tbody>
{rows(TRADE1, hi="1.05（现用）")}
</tbody></table>
<p>我选了 <b>1.05</b>：<span class="g">零误报</span>，代价是漏掉最弱的 1 张（<code>10_</code>，1.088）。</p>

<h3>为什么这样选</h3>
<ul>
<li><b>误报比漏报更贵。</b>产线上误报会直接导致停机、需要人工复检；
    而漏掉的那张是最微弱的改动（几乎不可见）。</li>
<li>而且 1.088 与 1.025 的差距在噪声范围内 —— 即使把阈值压到 1.0，
    也只是"这次碰巧检出"，下次运行可能又漏。不是稳定收益。</li>
</ul>

<h3>如果你想两者都要，有更好的办法</h3>
<p>不要靠调阈值，而是<b>把"深色纤维"这类可接受变异收进库</b>：</p>
<ol>
<li>如果产线上这类深色纤维<b>经常出现</b> → 收集 20~30 张这样的样本放进 OK 集重新建库。
    库见过之后，这类图分数会掉到正常区间，阈值就能降回 0.9 左右，灵敏度也回来了。</li>
<li>如果这类纤维<b>极其罕见</b>（比如 372 张里只有 1 张）→ 维持阈值 1.05 即可，
    偶尔漏掉一个几乎看不见的浅痕，可以接受。</li>
</ol>
<p class="k">这个判断需要你结合产品标准来定 —— 深色纤维到底是"常见且可接受"还是"偶发"？</p>
</div>

<div class="card">
<h2>四、相机2 的阈值权衡（无重叠，宽松）</h2>
<table><thead><tr><th>阈值</th><th>OK 误报</th><th>NG 检出</th><th>漏报</th><th>说明</th></tr></thead><tbody>
{rows(TRADE2, hi="0.84（现用）")}
</tbody></table>
<p>相机2 干净得多：正常上界 <b>0.772</b>，最弱可检出 NG <b>0.928</b>，
<span class="g">分离窗口 0.156</span>。阈值在 <b>0.78 ~ 0.90</b> 区间内都是零误报且检出 3/4，
有真正的余量。选 <b>0.84</b>（= mean+6σ）。</p>
</div>

<div class="card">
<h2>五、NG 检出复测（新模型）</h2>
<div class="grid">
<div>
<h3>皮料相机1（阈值 1.05）</h3>
<pre>检出率      9/11 = 81.8%
（排除未修改的 29 后为 9/10）
定位命中    8/11 = 72.7%
OK 误报     0</pre>
</div>
<div>
<h3>皮料相机2（阈值 0.84）</h3>
<pre>检出率      3/4 = 75.0%
定位命中    3/4 = 75.0%
OK 误报     0</pre>
</div>
</div>
<p><b>合计 12/15 = 80%，定位 11/15 = 73.3%</b>（旧模型是 66.7%，略有提升）。
与重建前完全持平 —— <span class="k">说明重建没有损害检出能力。</span></p>
<p class="sm">注意：这 14 张 NG 仍是你在 OK 图上手画的线（对比度高），
所以 80% 是乐观上界。唯一一次真实缺陷验证是相机2 那张白色划痕
（得分 ≈2.9，是阈值的 3.4 倍），但那张已按你的要求删除。</p>
</div>

<div class="card">
<h2>六、当前可交付状态</h2>
<table><thead><tr><th>项目</th><th>状态</th></tr></thead><tbody>
<tr><td>记忆库</td><td class="g">已用全量数据重建（cam1 372 张 / cam2 371 张）</td></tr>
<tr><td>阈值</td><td class="g">已重新标定（cam1 1.05 / cam2 0.84），已写入检测工具</td></tr>
<tr><td>OK 图误报</td><td class="g">0（两个相机）</td></tr>
<tr><td>NG 检出</td><td>12/15 = 80%（手画线，乐观上界）</td></tr>
<tr><td>真实缺陷验证</td><td class="r">样本已被删除，目前无留存样本</td></tr>
<tr><td>多机台部署</td><td>已验证须每台独立建库</td></tr>
<tr><td>速度</td><td class="g">CPU 约 226 ms/张，占 1s 节拍 23%</td></tr>
</tbody></table>

<h3>下一步建议</h3>
<ol>
<li><b>决定相机1 的深色纤维策略</b>（常见→收进库；罕见→维持 1.05）。</li>
<li><b>后续遇到真实缺陷，保留样本</b>。相机2 那张白色划痕是最好的验证样本，
    删掉有点可惜 —— 建议以后这类图移到 NG 集留着，用于定期复测。</li>
<li><b>影子模式上线</b>：只记录判定不触发停机，跑一周看实际 NG 率，
    与人工复核结果对比，用现场数据做最终校准。</li>
</ol>
</div>

<div class="card">
<h2>附：复现命令</h2>
<pre># 重建（全量数据）
python train_custom.py --data ok_data --classname 皮料相机1 \\
  --out models/cam1 --roi 540,30,1780,1270
python train_custom.py --data ok_data --classname 皮料相机2 \\
  --out models/cam2 --roi 15,30,1400,1270

# 重新定阈值（双向交叉验证，无偏）
python validate_ok_batch.py

# NG 检出复测
python eval_ng2.py

# 手动检测工具
python inspect_tool.py        （或双击 启动检测工具.bat）</pre>
<p class="sm">产物：<code>testimgs/report_rebuild.html</code>（本报告）、
<code>models/cam1</code>、<code>models/cam2</code>（新模型）、
<code>models/cam1_139bak</code>、<code>models/cam2_143bak</code>（旧模型备份）。</p>
</div>

</div></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print(f"written: {OUT} ({len(doc)} chars)")
