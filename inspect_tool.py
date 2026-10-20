"""PatchCore 手动检测工具（本地网页版，零额外依赖）。

特点:
  - 单张检测：想测哪张点哪张
  - 文件夹逐步检测：选文件夹 -> 列表点选 / 上下切换 / 一键全检
  - 阈值可实时调整（改阈值不用重新推理，分数已缓存）
  - 显示原图(带ROI框) / 异常热图 / 叠加图
  - 结果可导出 CSV

启动:
    .venv/Scripts/python.exe inspect_tool.py
然后浏览器打开 http://127.0.0.1:8765

可调参数:
    --port 8765        监听端口
    --host 127.0.0.1   监听地址（仅本机，不要改成 0.0.0.0）
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import PIL.Image
import PIL.ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from infer_custom import LeatherInspector  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# 站点配置 —— 新相机在这里加一条即可
# ---------------------------------------------------------------------------
SITES = {
    "皮料相机1": dict(
        label="皮料相机1 (CAM1)",
        model=os.path.join(HERE, "models", "cam1"),
        onnx=os.path.join(HERE, "models_onnx", "皮料相机1"),
        roi=(540, 30, 1780, 1270),
        # 全量 372 张建库；正常上界 1.025（含可接受的深色纤维样本），
        # 故阈值取 1.05 —— 牺牲最弱的 1 张，换取零误报
        threshold=1.05,
    ),
    "皮料相机2": dict(
        label="皮料相机2 (CAM2)",
        model=os.path.join(HERE, "models", "cam2"),
        onnx=os.path.join(HERE, "models_onnx", "皮料相机2"),
        roi=(15, 30, 1400, 1270),
        # 全量 371 张建库（已删除误标的 348），正常上界 0.772，阈值取 0.85
        threshold=0.85,
    ),
}

MODEL_ROOT_TORCH = os.path.join(HERE, "models")
MODEL_ROOT_ONNX = os.path.join(HERE, "models_onnx")


def _read_model_meta(d):
    """从模型目录读 roi / threshold / 向量数（meta_v2.json 优先，其次 meta.json）。"""
    info = {}
    for fn in ("meta_v2.json", "meta.json"):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        if isinstance(j.get("roi"), list) and len(j["roi"]) == 4:
            info["roi"] = tuple(int(v) for v in j["roi"])
        if "threshold" in j:
            info["threshold"] = float(j["threshold"])
        if "n_bank" in j:
            info["n_bank"] = int(j["n_bank"])
        if "n_ok" in j:
            info["n_ok"] = int(j["n_ok"])
        if isinstance(j.get("holdout"), dict):
            info["holdout"] = j["holdout"]
        break
    return info


def discover_sites():
    """手写 SITES + 自动发现 models_onnx/ 下的每个模型目录。

    建库工具把模型输出到 `models_onnx/<名字>/`，这里扫一遍就能在界面下拉里
    直接切换，不用再回来改源码里的 SITES。以 `_` 开头的目录跳过（`_base`
    是骨干模板，不是可用模型）。
    """
    out = {k: dict(v) for k, v in SITES.items()}
    if not os.path.isdir(MODEL_ROOT_ONNX):
        return out
    for name in sorted(os.listdir(MODEL_ROOT_ONNX)):
        if name.startswith("_") or name in out:
            continue
        d = os.path.join(MODEL_ROOT_ONNX, name)
        if not os.path.isdir(d):
            continue
        has_new = os.path.exists(os.path.join(d, "patchcore_roi.onnx"))
        has_old = (os.path.exists(os.path.join(d, "model.onnx"))
                   and os.path.exists(os.path.join(d, "memory_bank.npy")))
        if not (has_new or has_old):
            continue
        info = _read_model_meta(d)
        bits = []
        if has_new:
            bits.append("一体化")
        if info.get("n_bank"):
            bits.append("%d 向量" % info["n_bank"])
        if "threshold" in info:
            bits.append("阈值 %.2f" % info["threshold"])
        out[name] = dict(
            label=("%s（%s）" % (name, "，".join(bits))) if bits else name,
            model=None,          # 自动发现的目录只认 ONNX，避免和旧 torch 模型混用
            onnx=d,
            roi=info.get("roi", (0, 0, 0, 0)),
            threshold=info.get("threshold", 0.5),
            n_bank=info.get("n_bank"),
            auto=True,
        )
    return out


def get_site_cfg(site):
    sites = discover_sites()
    if site not in sites:
        raise KeyError("未知模型：%s（可用：%s）" % (site, "、".join(sites)))
    return sites[site]


# 推理后端：
#   torch —— 原实现：特征提取走 GPU，FAISS 检索走 CPU，约 104ms
#   onnx  —— 纯 CPU，不需要 GPU / FAISS / PyTorch，适合无独显工控机，约 300ms
BACKENDS = ("torch", "onnx")
_active_backend = "torch"

DISP = 460          # 显示区边长(px)

_models = {}        # (backend, site, roi) -> 后端实例
_cache = {}         # (backend, site, path, roi) -> dict
_jobs = {}          # job_id -> dict(done,total,current,state)
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 模型加载与推理（两个后端统一接口 predict(path) -> (score, mask_224)）
# ---------------------------------------------------------------------------
def onnx_available(site=None):
    sites = discover_sites()
    for s in ([site] if site else list(sites)):
        cfg = sites.get(s)
        d = (cfg or {}).get("onnx")
        if not d or not os.path.isdir(d):
            return False
        # 一体化模型（建库工具产物）或旧的两文件模型，都算可用
        if os.path.exists(os.path.join(d, "patchcore_roi.onnx")):
            continue
        if (os.path.exists(os.path.join(d, "model.onnx"))
                and os.path.exists(os.path.join(d, "memory_bank.npy"))):
            continue
        return False
    return True


def get_backend(site, roi_override=None, backend=None):
    """按后端类型惰性加载并常驻内存。"""
    backend = backend or _active_backend
    cfg = get_site_cfg(site)
    roi = tuple(roi_override or cfg["roi"])
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("%s 的 ROI 无效 %s，请检查模型目录里的 meta.json" % (site, roi))
    key = (backend, site, tuple(roi))
    with _lock:
        if key in _models:
            return _models[key]
        if backend == "onnx":
            from onnx_infer import OnnxPatchCore
            if not cfg.get("onnx") or not os.path.isdir(cfg["onnx"]):
                raise FileNotFoundError(
                    f"ONNX 模型不存在: {cfg.get('onnx')}，请先运行 export_onnx.py")
            print(f"[加载] ONNX {site}  dir={cfg['onnx']}  roi={roi}")
            det = OnnxPatchCore(cfg["onnx"], threshold=cfg["threshold"], roi=roi)
        else:
            if not cfg.get("model") or not os.path.isdir(cfg["model"]):
                raise FileNotFoundError(
                    f"{site} 没有 torch 模型（只有 ONNX），请把后端切到 onnx")
            print(f"[加载] torch {site}  model={cfg['model']}  roi={roi}")
            det = LeatherInspector(cfg["model"], roi=roi)
        _models[key] = det
        return det


def get_inspector(site, roi_override=None):
    """兼容旧调用（固定 torch 后端）。"""
    return get_backend(site, roi_override, "torch")


_JET = None


def _jet(norm):
    """把 0~1 的浮点图映射成 RGB（jet 色表，惰性加载并缓存）。"""
    global _JET
    if _JET is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import colormaps
        _JET = colormaps["jet"]
    rgba = _JET(np.clip(norm, 0, 1))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def render_original(im, roi, size=DISP):
    """原图（整图缩放）+ 红框标出 ROI。只作定位参考，用 BILINEAR 求快。"""
    full = im.resize((size, size), PIL.Image.BILINEAR)
    d = PIL.ImageDraw.Draw(full)
    sx, sy = size / im.size[0], size / im.size[1]
    x0, y0, x1, y1 = roi
    d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                outline=(230, 60, 60), width=2)
    return full


def render_roi_crop(im, roi, size=DISP):
    """ROI 裁剪后的料面。这是看细节的主视图，保留高质量重采样。"""
    return im.crop(roi).resize((size, size), PIL.Image.LANCZOS)


def render_heat(mask, size=DISP):
    """异常热图（单色）。"""
    m = np.asarray(mask, dtype=np.float32)
    lo, hi = float(m.min()), float(m.max())
    norm = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    rgb = _jet(norm)
    return PIL.Image.fromarray(rgb).resize((size, size), PIL.Image.BILINEAR)


def render_overlay_from(base_crop, heat_img, alpha=0.5):
    """热图叠加到已渲染好的料面图上（复用，避免重复解码/缩放）。"""
    return PIL.Image.blend(base_crop, heat_img, alpha)


def img_data_url(img, kind="photo"):
    """转 data URL。

    性能要点（实测差异很大）：
      - PNG 加 optimize=True 时单张编码约 60ms；4 张合计 244ms，是主要瓶颈
      - 照片类用 JPEG(q85) 约 10~15ms，且体积更小
      - 热图有硬色边，用低压缩 PNG 保住色阶
    """
    buf = io.BytesIO()
    if kind == "photo":
        img.save(buf, format="JPEG", quality=85, subsampling=2, optimize=False)
        head = "data:image/jpeg;base64,"
    else:
        img.save(buf, format="PNG", compress_level=1)
        head = "data:image/png;base64,"
    return head + base64.b64encode(buf.getvalue()).decode()


def png_data_url(img):
    """兼容旧调用。"""
    return img_data_url(img, kind="photo")


def detect(site, path, roi_override=None, force=False, backend=None):
    """推理单张，结果缓存（缓存键含后端，切换后端不会串结果）。"""
    cfg = get_site_cfg(site)
    roi = tuple(roi_override or cfg["roi"])
    backend = backend or _active_backend
    key = (backend, site, path, roi)
    with _lock:
        if not force and key in _cache:
            return _cache[key]

    det = get_backend(site, roi, backend)
    t0 = time.perf_counter()
    score, mask = det.predict(path)
    ms = (time.perf_counter() - t0) * 1000

    # 响应集中度：最高 1% 像素贡献的能量占比
    flat = np.sort(np.asarray(mask).ravel())
    top = flat[int(len(flat) * 0.99):]
    conc = float(top.sum() / max(flat.sum(), 1e-9)) * 100

    # 原图只解码一次，四个视图全部复用 —— 之前每次 render 都重新 open+decode
    im = PIL.Image.open(path).convert("RGB")
    t1 = time.perf_counter()
    img_full = render_original(im, roi)
    img_crop = render_roi_crop(im, roi)
    img_heat = render_heat(mask)
    img_ovl = render_overlay_from(img_crop, img_heat)
    render_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    rec = dict(
        score=round(float(score), 4),
        ms=round(ms, 1),
        concentration=round(conc, 1),
        backend=backend,
        original=img_data_url(img_full, "photo"),
        crop=img_data_url(img_crop, "photo"),
        heat=img_data_url(img_heat, "heat"),
        overlay=img_data_url(img_ovl, "photo"),
    )
    enc_ms = (time.perf_counter() - t2) * 1000
    rec["render_ms"] = round(render_ms, 1)
    rec["encode_ms"] = round(enc_ms, 1)
    with _lock:
        _cache[key] = rec
    return rec


# ---------------------------------------------------------------------------
# 文件浏览
# ---------------------------------------------------------------------------
def list_drives():
    out = []
    for c in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        p = f"{c}:/"
        if os.path.exists(p):
            out.append(p)
    # 常用起始位置
    for p in ["C:/Users/Administrator/Desktop", "C:/Users/Administrator"]:
        if os.path.exists(p) and p not in out:
            out.insert(0, p)
    return out


def browse(path):
    """返回该目录下的子目录与图片。"""
    if not path or not os.path.isdir(path):
        return dict(path="", parent="", dirs=list_drives(), images=[])
    path = os.path.normpath(path).replace("\\", "/")
    dirs, images = [], []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                dirs.append(name)
            elif os.path.splitext(name)[1].lower() in IMG_EXT:
                images.append(name)
    except PermissionError:
        pass
    parent = os.path.dirname(path.rstrip("/")) or ""
    return dict(path=path, parent=parent.replace("\\", "/"), dirs=dirs, images=images)


def guess_site(folder):
    """按文件夹名猜该用哪个相机模型。"""
    s = (folder or "").lower()
    if "相机1" in s or "cam1" in s:
        return "皮料相机1"
    if "相机2" in s or "cam2" in s:
        return "皮料相机2"
    return None


# ---------------------------------------------------------------------------
# 后台批量检测
# ---------------------------------------------------------------------------
def start_job(site, folder, roi_override=None, backend=None):
    # 批量期间固定后端，避免用户中途切换导致同一批结果混杂
    backend = backend or _active_backend
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXT
    )
    job_id = str(int(time.time() * 1000))
    job = dict(done=0, total=len(files), state="running", backend=backend,
               folder=folder, site=site, results={}, order=files)
    _jobs[job_id] = job

    def worker():
        for i, f in enumerate(files):
            p = os.path.join(folder, f)
            try:
                r = detect(site, p, roi_override, backend=backend)
                job["results"][f] = r["score"]
            except Exception as e:
                job["results"][f] = None
                print(f"[批量] {f} 失败: {e}")
            job["done"] = i + 1
        job["state"] = "done"
        print(f"[批量] 完成 {job['done']}/{job['total']}  后端={backend}")

    threading.Thread(target=worker, daemon=True).start()
    return job_id, job


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "PatchCoreInspect/1.0"

    def log_message(self, fmt, *args):
        pass      # 静音，避免刷屏

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._send(code, dict(error=str(msg)))

    def do_GET(self):
        try:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            route = u.path

            if route == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")

            if route == "/api/sites":
                return self._send(200, {
                    k: dict(label=v["label"], roi=list(v["roi"]),
                            threshold=v["threshold"],
                            torch_ok=bool(v.get("model")) and os.path.isdir(v["model"]),
                            onnx_ok=onnx_available(k),
                            n_bank=v.get("n_bank"),
                            auto=bool(v.get("auto")))
                    for k, v in discover_sites().items()
                })

            if route == "/api/backend":
                # 若已加载 ONNX 会话，如实报告实际生效的 provider
                # （请求 CUDA 但版本不匹配时 ORT 会静默回退 CPU）
                prov = None
                with _lock:
                    for (be, _s, _r), obj in _models.items():
                        if be == "onnx" and hasattr(obj, "providers_actual"):
                            prov = list(obj.providers_actual)
                            break
                return self._send(200, dict(
                    active=_active_backend,
                    available=[b for b in BACKENDS
                               if b == "torch" or onnx_available()],
                    onnx_ready=onnx_available(),
                    onnx_provider=prov,
                ))

            if route == "/api/browse":
                return self._send(200, browse(q.get("path", "")))

            if route == "/api/progress":
                jid = q.get("job", "")
                j = _jobs.get(jid)
                if not j:
                    return self._err("job 不存在", 404)
                return self._send(200, dict(
                    done=j["done"], total=j["total"], state=j["state"],
                    results=j["results"], order=j["order"]))

            return self._err("未知路径", 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 浏览器刷新/关闭、或请求被中断时的正常现象，不必打堆栈
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                return self._err(e, 500)
            except Exception:
                pass

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            u = urlparse(self.path)

            if u.path == "/api/detect":
                site = req["site"]
                path = req["path"]
                if not os.path.isfile(path):
                    return self._err(f"文件不存在: {path}")
                roi = req.get("roi")
                r = detect(site, path, roi, force=bool(req.get("force")))
                cfg = get_site_cfg(site)
                thr = float(req.get("threshold", cfg["threshold"]))
                r = dict(r)
                r["verdict"] = "NG" if r["score"] >= thr else "OK"
                r["threshold"] = thr
                r["file"] = os.path.basename(path)
                return self._send(200, r)

            if u.path == "/api/detect_folder":
                site = req["site"]
                folder = req["folder"]
                if not os.path.isdir(folder):
                    return self._err(f"目录不存在: {folder}")
                jid, job = start_job(site, folder, req.get("roi"))
                return self._send(200, dict(job=jid, total=job["total"]))

            if u.path == "/api/clear_cache":
                with _lock:
                    _cache.clear()
                return self._send(200, dict(ok=True))

            if u.path == "/api/backend":
                global _active_backend
                b = req.get("backend")
                if b not in BACKENDS:
                    return self._err(f"未知后端: {b}")
                if b == "onnx" and not onnx_available():
                    return self._err("ONNX 模型未导出，请先运行 export_onnx.py")
                _active_backend = b
                print(f"[后端] 切换为 {b}", flush=True)
                return self._send(200, dict(active=_active_backend))

            return self._err("未知路径", 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 浏览器刷新/关闭、或请求被中断时的正常现象，不必打堆栈
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                return self._err(e, 500)
            except Exception:
                pass


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>PatchCore 手动检测工具</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         background:#f5f6f8; color:#1c1f23; font-size:13px; }
  .bar { background:#fff; border-bottom:1px solid #e3e6ea; padding:10px 16px;
         display:flex; align-items:center; gap:14px; flex-wrap:wrap;
         position:sticky; top:0; z-index:10; }
  .bar h1 { font-size:15px; margin:0; font-weight:500; }
  .bar label { color:#5a6472; }
  select,input[type=number],input[type=text] { font:inherit; padding:5px 8px;
         border:1px solid #cfd4da; border-radius:6px; background:#fff; }
  input[type=range] { width:150px; vertical-align:middle; }
  button { font:inherit; padding:6px 12px; border:1px solid #cfd4da; border-radius:6px;
         background:#fff; cursor:pointer; white-space:nowrap; }
  button:hover { background:#f0f2f5; }
  button.primary { background:#1a56c4; border-color:#1a56c4; color:#fff; }
  button.primary:hover { background:#164ba8; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .wrap { display:flex; height:calc(100vh - 118px); }
  .side { width:290px; border-right:1px solid #e3e6ea; background:#fff;
          display:flex; flex-direction:column; }
  .crumbs { padding:8px 12px; border-bottom:1px solid #eceff2; color:#5a6472;
            font-size:12px; word-break:break-all; }
  .tree { flex:1; overflow:auto; padding:6px 0; }
  .item { padding:5px 12px; cursor:pointer; display:flex; justify-content:space-between;
          gap:8px; align-items:center; }
  .item:hover { background:#f0f2f5; }
  .item.sel { background:#e8f0fe; }
  .item .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .score { font-family:Consolas,monospace; font-size:11.5px; padding:1px 6px;
           border-radius:4px; background:#eef1f4; color:#5a6472; flex:none; }
  .score.ng { background:#ffebee; color:#c62828; font-weight:700; }
  .score.ok { background:#e8f5e9; color:#2e7d32; }
  .main { flex:1; overflow:auto; padding:16px; }
  .row { display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; }
  .card { background:#fff; border:1px solid #e3e6ea; border-radius:10px; padding:12px; }
  .card h3 { margin:0 0 8px; font-size:12px; font-weight:500; color:#5a6472; }
  .card img { display:block; border-radius:6px; max-width:100%; }
  .badge { font-size:34px; font-weight:700; line-height:1.1; }
  .badge.ok { color:#2e7d32; } .badge.ng { color:#c62828; }
  .meta { margin-top:10px; line-height:1.9; }
  .meta b { font-family:Consolas,monospace; }
  .hint { color:#8a929c; font-size:12px; }
  .prog { height:6px; background:#eceff2; border-radius:3px; overflow:hidden;
          margin-top:8px; }
  .prog i { display:block; height:100%; background:#1a56c4; width:0; }
  .empty { color:#8a929c; padding:40px; text-align:center; }
  .zoom { display:flex; gap:4px; }
  .zoom button { padding:3px 9px; font-size:12px; }
  .zoom button.on { background:#e8f0fe; border-color:#1a56c4; color:#1a56c4; }
</style></head><body>

<div class="bar">
  <h1>PatchCore 手动检测</h1>
  <label>模型</label>
  <select id="site"></select>
  <label>后端</label>
  <select id="backend"></select>
  <label>阈值</label>
  <input type="range" id="thrR" min="0" max="300" value="91">
  <input type="number" id="thr" step="0.01" style="width:74px">
  <label>显示</label>
  <div class="zoom">
    <button data-z="360">小</button>
    <button data-z="460" class="on">中</button>
    <button data-z="640">大</button>
  </div>
  <span style="flex:1"></span>
  <button id="btnDetect" class="primary">检测当前</button>
  <button id="btnAll">检测整个文件夹</button>
  <button id="btnCsv">导出 CSV</button>
  <button id="btnClear">清空结果</button>
</div>

<div class="wrap">
  <div class="side">
    <div class="crumbs" id="crumbs">未选择文件夹</div>
    <div style="padding:8px 12px;border-bottom:1px solid #eceff2;display:flex;gap:6px;flex-wrap:wrap">
      <button onclick="goParent()">上级</button>
      <button onclick="browseTo('')">盘符</button>
      <button onclick="pickFolder()">选用此文件夹</button>
    </div>
    <div class="tree" id="tree"></div>
  </div>
  <div class="main" id="main">
    <div class="empty">从左侧选择文件夹，或直接点图片开始检测</div>
  </div>
</div>

<script>
let SITES={}, curSite="", curDir="", curFile="", mode="browse", files=[], idx=-1, cur=null, disp=460;
const $=id=>document.getElementById(id);

async function api(path, opt){ const r=await fetch(path,opt); const j=await r.json();
  if(j.error) throw new Error(j.error); return j; }
async function post(path, body){
  return api(path,{method:"POST",headers:{"Content-Type":"application/json"},
                   body:JSON.stringify(body)}); }

async function init(){
  SITES=await api("/api/sites");
  const sel=$("site");
  sel.innerHTML="";
  for(const k in SITES){
    const o=document.createElement("option"); o.value=k;
    const be=(SITES[k].onnx_ok?"ONNX":"")+(SITES[k].onnx_ok&&SITES[k].torch_ok?"/":"")+(SITES[k].torch_ok?"torch":"");
    o.textContent=SITES[k].label+(be?"  ["+be+"]":"  (缺模型)");
    sel.appendChild(o);
  }
  curSite=Object.keys(SITES)[0];
  syncThr();
  sel.onchange=()=>{ curSite=sel.value; syncThr(); if(curFile) redetect(); renderScores(); };

  await initBackend();
  await browseTo("");
}

/* ---- 推理后端切换 ---- */
async function initBackend(){
  const b=await api("/api/backend");
  const el=$("backend"); el.innerHTML="";
  for(const k of b.available){
    const o=document.createElement("option"); o.value=k;
    o.textContent = k==="onnx" ? "ONNX (CPU · 无独显)" : "PyTorch (GPU · 快)";
    el.appendChild(o);
  }
  el.value=b.active;
  if(!b.onnx_ready && !b.available.includes("onnx")){
    const o=document.createElement("option"); o.disabled=true;
    o.textContent="ONNX (未导出)"; el.appendChild(o);
  }
  el.onchange=async()=>{
    try{
      await post("/api/backend",{backend:el.value});
      // 换后端后分数有微小差异，旧分数作废
      SCORES={}; window.__local={};
      renderScores();
      if(curFile) await redetect();
    }catch(e){
      alert("切换失败："+e.message);
      el.value=BE_saved;
    }
    BE_saved=el.value;
  };
  BE_saved=b.active;
}
let BE_saved="torch";

function syncThr(){
  const t=SITES[curSite].threshold;
  $("thr").value=t; $("thrR").value=Math.round(t*100);
}
$("thrR").oninput=()=>{ $("thr").value=($("thrR").value/100).toFixed(2);
  if(cur) verdict(); renderScores(); };
$("thr").oninput=()=>{ $("thrR").value=Math.round(parseFloat($("thr").value||0)*100);
  if(cur) verdict(); renderScores(); };
document.querySelectorAll(".zoom button").forEach(b=>{
  b.onclick=()=>{ document.querySelectorAll(".zoom button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on"); disp=parseInt(b.dataset.z); if(curFile) redetect(); };
});

async function browseTo(path){
  const r=await api("/api/browse?path="+encodeURIComponent(path));
  curDir=r.path; mode="browse"; files=[]; idx=-1;
  $("crumbs").textContent = r.path || "选择盘符或常用位置";
  const t=$("tree"); t.innerHTML="";
  if(!r.path){
    r.dirs.forEach(d=>t.appendChild(item(d,()=>browseTo(d),"📁")));
    return;
  }
  r.dirs.forEach(d=>t.appendChild(item(d,()=>browseTo(join(r.path,d)),"📁")));
  r.images.forEach(f=>t.appendChild(item(f,()=>openSingle(f),"🖼")));
  if(r.images.length) {
    t.insertAdjacentHTML("afterbegin",
      `<div class="hint" style="padding:6px 12px">${r.images.length} 张图片 · 点任意一张即可单张检测</div>`);
  }
}
function join(a,b){ return a.replace(/\/$/,"")+"/"+b; }
function item(name,onclick,ic){
  const d=document.createElement("div"); d.className="item";
  d.innerHTML=`<span class="nm">${ic} ${name}</span>`;
  d.onclick=onclick; return d;
}
function goParent(){ if(!curDir) return;
  const p=curDir.replace(/\/[^\/]+$/,"")||""; browseTo(p); }

/* 单张检测：不依赖文件夹列表，想测哪张点哪张 */
async function openSingle(f){
  if(!curDir) return;
  mode="single"; idx=-1; curFile=f;
  const g=guessSite(curDir);
  if(g && g!==curSite){ curSite=g; $("site").value=g; syncThr(); }
  $("main").innerHTML=`<div class="empty">检测中… ${f}</div>`;
  try{
    await detectPath(join(curDir,f), f);
    draw();
  }catch(e){ $("main").innerHTML=`<div class="empty">出错：${e.message}</div>`; }
}

/* 统一入口：走服务端推理，结果同时进缓存与分数表 */
async function detectPath(path, fname){
  const thr=parseFloat($("thr").value)||0.9;
  const r=await post("/api/detect",
    {site:curSite,path:path,threshold:thr,force:false});
  cur=r; curFile=fname||r.file;
  SCORES[curFile]=r.score;
  if(!window.__local) window.__local={};
  window.__local[curFile+curSite]=r.score;
  return r;
}

async function pickFolder(){
  if(!curDir){ alert("请先进入一个文件夹"); return; }
  const r=await api("/api/browse?path="+encodeURIComponent(curDir));
  files=r.images||[];
  if(!files.length){ alert("该文件夹没有图片"); return; }
  mode="folder"; idx=-1;
  const g=guessSite(curDir);
  if(g && g!==curSite){ curSite=g; $("site").value=g; syncThr(); }
  renderList();
}

function guessSite(p){
  const s=(p||"").toLowerCase();
  if(s.includes("相机1")||s.includes("cam1")) return "皮料相机1";
  if(s.includes("相机2")||s.includes("cam2")) return "皮料相机2";
  return null;
}

function renderList(){
  const t=$("tree"); t.innerHTML="";
  t.insertAdjacentHTML("afterbegin",
    `<div class="hint" style="padding:6px 12px">
       ${files.length} 张 · 键盘 ↑↓ 切换，回车重检
     </div>`);
  files.forEach((f,i)=>{
    const d=document.createElement("div");
    d.className="item"+(i===idx?" sel":"");
    d.id="it"+i;
    d.innerHTML=`<span class="nm">${i+1}. ${f}</span><span class="score" id="sc${i}">-</span>`;
    d.onclick=()=>openIndex(i);
    t.appendChild(d);
  });
}

/* 只要当前文件夹在检测，且该文件有分数，就刷新左侧徽标 */
function renderScores(){
  const thr=parseFloat($("thr").value)||0.9;
  const paint=(el,f)=>{
    if(!el) return;
    const s=(window.__local||{})[f+curSite];
    if(s===undefined||s===null){ el.textContent="-"; el.className="score"; return; }
    el.textContent=s.toFixed(3);
    el.className="score "+(s>=thr?"ng":"ok");
  };
  if(mode==="folder"){
    files.forEach((f,i)=>paint($("sc"+i), f));
  }else if(mode==="single"){
    document.querySelectorAll("#tree .item").forEach(d=>{
      const nm=d.querySelector(".nm").textContent.replace(/^🖼\s*/,"");
      let sc=d.querySelector(".score");
      if(!sc){ sc=document.createElement("span"); d.appendChild(sc); }
      paint(sc, nm);
    });
  }
}

let SCORES={};

async function openIndex(i){
  if(mode!=="folder"){ return; }
  if(i<0||i>=files.length) return;
  idx=i;
  const f=files[i];
  renderList();
  $("btnDetect").disabled=true;
  $("main").innerHTML=`<div class="empty">检测中… ${f}</div>`;
  try{
    await detectPath(join(curDir,f), f);
    draw();
  }catch(e){ $("main").innerHTML=`<div class="empty">出错：${e.message}</div>`; }
  $("btnDetect").disabled=false;
}

function verdict(){
  if(!cur) return null;
  const thr=parseFloat($("thr").value)||0;
  return cur.score>=thr?"NG":"OK";
}
function draw(){
  const v=verdict();
  const cls=v==="NG"?"ng":"ok";
  $("main").innerHTML=`
   <div class="row">
    <div class="card"><h3>原图（红框 = ROI 取点区）</h3>
      <img src="${cur.original}" id="im0"></div>
    <div class="card"><h3>ROI 料面</h3>
      <img src="${cur.crop}" id="im1"></div>
   </div>
   <div class="row" style="margin-top:16px">
    <div class="card"><h3>异常热图</h3>
      <img src="${cur.heat}" id="im2"></div>
    <div class="card"><h3>叠加</h3>
      <img src="${cur.overlay}" id="im3"></div>
    <div class="card" style="min-width:230px">
      <h3>判定结果</h3>
      <div class="badge ${cls}">${v}</div>
      <div class="meta">
        文件　<b>${cur.file}</b><br>
        分数　<b>${cur.score.toFixed(3)}</b><br>
        阈值　<b>${(parseFloat($("thr").value)).toFixed(2)}</b><br>
        响应集中度　<b>${cur.concentration}%</b><br>
        推理耗时　<b>${cur.ms} ms</b><br>
        后端　<b>${cur.backend==="onnx"?"ONNX (CPU)":"PyTorch"}</b>
      </div>
      <div class="hint" style="margin-top:8px">
        集中度低(&lt;10%) = 局部缺陷定位<br>
        集中度高(&gt;50%)但分数低 = 纹理噪声，不判NG
      </div>
      ${mode==="folder"?`<div class="meta">进度 <b>${idx+1} / ${files.length}</b></div>
        <div style="display:flex;gap:6px;margin-top:8px">
          <button onclick="openIndex(${idx-1})" ${idx<=0?"disabled":""}>← 上一张</button>
          <button onclick="openIndex(${idx+1})" ${idx>=files.length-1?"disabled":""}>下一张 →</button>
        </div>`:""}
    </div>
   </div>`;
}

/* 重检当前图（缩放变化 / 手动点「检测当前」时用，force 跳过缓存） */
async function redetect(){
  if(!curFile){ alert("请先在左侧点一张图片"); return; }
  $("btnDetect").disabled=true;
  try{
    const thr=parseFloat($("thr").value)||0.9;
    const r=await post("/api/detect",
      {site:curSite,path:join(curDir,curFile),threshold:thr,force:true});
    cur=r;
    SCORES[curFile]=r.score;
    if(!window.__local) window.__local={};
    window.__local[curFile+curSite]=r.score;
    draw(); renderScores();
  }catch(e){ alert("出错："+e.message); }
  $("btnDetect").disabled=false;
}

$("btnDetect").onclick=redetect;

$("btnAll").onclick=async()=>{
  if(mode!=="folder"){ alert("请先点左侧的「选用此文件夹」"); return; }
  $("main").innerHTML=`<div class="empty">正在检测 ${files.length} 张…<div class="prog"><i id="pi"></i></div></div>`;
  const r=await post("/api/detect_folder",{site:curSite,folder:curDir});
  const jid=r.job;
  const t=setInterval(async()=>{
    const p=await api("/api/progress?job="+jid);
    const el=$("pi"); if(el) el.style.width=(p.done/p.total*100)+"%";
    Object.assign(SCORES,p.results);
    for(const k in p.results) window.__local[k+curSite]=p.results[k];
    renderScores();
    if(p.state==="done"){
      clearInterval(t);
      $("main").innerHTML=`<div class="empty">全部完成 ${p.done}/${p.total}，
        点左侧图片查看细节</div>`;
    }
  },400);
};

$("btnCsv").onclick=()=>{
  if(mode!=="folder"){ alert("请先选用一个文件夹"); return; }
  const thr=parseFloat($("thr").value)||0.9;
  const rows=[["filename","score","verdict","threshold","site"]];
  files.forEach(f=>{
    const s=(f in SCORES)?SCORES[f]:window.__local?.[f+curSite];
    rows.push([f, s===undefined?"":s.toFixed(4),
      s===undefined?"":(s>=thr?"NG":"OK"), thr, curSite]);
  });
  const csv="\ufeff"+rows.map(r=>r.join(",")).join("\n");
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download="inspect_result.csv"; a.click();
};

$("btnClear").onclick=async()=>{
  await post("/api/clear_cache",{});
  SCORES={}; window.__local={}; renderScores();
};

document.addEventListener("keydown",e=>{
  if(mode!=="folder") {
    if(e.key==="Enter"){ e.preventDefault(); redetect(); }
    return;
  }
  if(e.key==="ArrowDown"||e.key==="ArrowRight"){ e.preventDefault(); openIndex(idx+1); }
  if(e.key==="ArrowUp"||e.key==="ArrowLeft"){ e.preventDefault(); openIndex(idx-1); }
  if(e.key==="Enter"){ e.preventDefault(); redetect(); }
});

init().catch(e=>{ document.getElementById("main").innerHTML=
  `<div class="empty">初始化失败：${e.message}</div>`; });
</script></body></html>
"""


def _probe_existing(host, port, timeout=0.6):
    """探测该端口上是否已有本工具在运行。

    Windows 上 HTTPServer 默认 allow_reuse_address=1，两个进程能同时绑同一端口，
    所以不能靠 bind 失败来发现重复启动 —— 必须主动探测。
    """
    import urllib.request
    import json as _json
    try:
        with urllib.request.urlopen(
                f"http://{host}:{port}/api/sites", timeout=timeout) as r:
            data = _json.loads(r.read().decode("utf-8"))
        # 只有返回本工具的结构才算"已有实例"
        if isinstance(data, dict) and any(k in data for k in SITES):
            return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true",
                    help="服务就绪后自动打开浏览器（推荐，避免网页先打开导致加载失败）")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"

    # 已有实例在跑 -> 直接复用，不再起第二个（否则 Windows 上会双绑定抢请求）
    if _probe_existing(args.host, args.port):
        print("=" * 62, flush=True)
        print(f"  检测到已有实例正在运行：{url}", flush=True)
        print("  直接打开浏览器，不再重复启动。", flush=True)
        print("=" * 62, flush=True)
        if args.open or True:
            import webbrowser
            try:
                webbrowser.open(url)
                print(f"  已在浏览器打开 {url}", flush=True)
            except Exception:
                print(f"  请手动访问 {url}", flush=True)
        try:
            input("\n按回车关闭本窗口（不影响已在运行的服务）...")
        except Exception:
            pass
        return

    missing = [k for k, v in SITES.items() if not os.path.isdir(v["model"])]
    if missing:
        print(f"警告：以下站点缺少模型目录 {missing}", flush=True)
        print("      请先用 train_custom.py 建库，或修改 SITES 配置。", flush=True)

    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        print("=" * 62, flush=True)
        print(f"  启动失败：端口 {args.port} 无法绑定（{e}）", flush=True)
        print("  可能该端口被其他程序占用。换个端口试试：", flush=True)
        print(f"      python inspect_tool.py --port 8766 --open", flush=True)
        print("=" * 62, flush=True)
        try:
            input("按回车退出...")
        except Exception:
            pass
        return

    print("=" * 62, flush=True)
    print("  PatchCore 手动检测工具已启动", flush=True)
    print(f"  请在浏览器打开:  {url}", flush=True)
    print("=" * 62, flush=True)
    for k, v in SITES.items():
        print(f"  {k}: threshold={v['threshold']}  roi={v['roi']}", flush=True)
    print("\n  Ctrl+C 或关闭本窗口即停止", flush=True)

    # 关键：端口已经绑定成功后才开浏览器，避免"网页先打开、服务还没起来"
    if args.open:
        import threading
        import webbrowser

        def _open():
            time.sleep(0.4)
            try:
                webbrowser.open(url)
                print(f"  已在浏览器打开 {url}", flush=True)
            except Exception as ex:
                print(f"  自动打开浏览器失败（{ex}），请手动访问 {url}", flush=True)

        threading.Thread(target=_open, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", flush=True)


if __name__ == "__main__":
    main()
