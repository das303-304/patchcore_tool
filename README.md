# PatchCore Tool

工业视觉用的 **PatchCore 无监督缺陷检测**内核与建库工具：只用 OK 图建记忆库，
检测未见过的不良，不需要标注、不需要训练。

面向皮料/织物等纹理类表面缺陷检测，已在产线验证。支持多相机独立建库、
阈值标定、ONNX 导出与部署验证。

---

## 目录结构

```
.
├── build_tool.py              建库主程序（含 Web UI 服务）
├── build_tool_ui.html         建库界面（单页，由 build_tool.py 提供）
├── build_tool_config.json     上次建库的参数（相机/ROI/目标库规模/骨干）
├── inspect_tool.py            模型检查：骨干、库规模、阈值、meta 一致性
├── onnx_infer.py              ONNX 推理与分数验证
├── infer_custom.py            单图推理调试
├── requirements-frozen.txt    依赖锁定
├── setup_env.sh               环境初始化脚本
├── 启动建库工具.bat            双击启动建库 Web UI
├── 启动检测工具.bat            双击启动检测界面
├── tools/                     开发期分析脚本（47 个，见文末清单）
└── docs/
```

---

## 快速开始

环境要求：Python 3.9+；建库建议有 NVIDIA GPU（CPU 也能建，只是慢）。

```bash
pip install -r requirements-frozen.txt

# Windows 下直接双击
启动建库工具.bat     # 建库界面，默认 http://127.0.0.1:8766
```

建库流程：

1. 准备一个目录，只放**正常（OK）样本**，建议 100 张以上
2. 在界面里选相机通道、框 ROI、设置目标库规模与骨干
3. 建库 → 自动导出 `patchcore_roi.onnx` + `meta_v2.json`
4. 用 `启动检测工具.bat` 或 `onnx_infer.py` 验证效果
5. 用 `tools/calibrate_*.py` 标定阈值

---

## 模型形态

当前是 **v2 形态**（`tools/build_v2.py`）：**ROI 不固化在图里**。

```
外部按 ROI 整数裁剪 → 送进 ONNX
  图内：缩放到 256 → 中心裁剪 224 → 归一化 → 骨干提特征 → 与记忆库算 KNN 距离
  输出：score（异常分）+ anomaly_map（224×224 热图）
```

这样设计的原因：ROI 随工位调整，固化在模型里会导致换 ROI 就得重建库。
代价是**调用方必须保证裁剪方式与建库时一致**（整数裁剪，不要插值缩放）。

> 若调用方额外做了重采样，特征就和建库时对不上。实测用非抗锯齿的 LINEAR 代替
> 建库时的抗锯齿缩放，正常料的分数会翻倍并被判成 NG。

### 两个文件必须成对

| 文件 | 内容 |
|---|---|
| `patchcore_roi.onnx` | 模型本体（骨干 + 记忆库） |
| `meta_v2.json` | ROI、阈值、正常均值、骨干名、库规模、生成时间戳 |

只换其中一个会导致阈值/ROI 错位。`inspect_tool.py` 可以检查两者是否一致。

---

## 关键概念

**为什么每台相机必须独立建库**
记忆库记录的是"这批像素长什么样"。换一台相机、换一次打光、动一次物距，
成像条件就变了，拿 A 机的库跑 B 机必然成片误报。
`tools/crosscam_validate.py` 与 `tools/test_cross_machine.py` 就是验证这件事的实验，
结论是**归一化只能缓解、不能消除**（见 `tools/test_norm_verdict.py`）。

**库规模（coreset 目标数）**
记忆库越大越准，但也越慢、越占内存。实测 3000 → 6000 时 OK 侧余量从 15.7% 提升到 17.5%，
NG 检出率不变。`tools/bench_tradeoff.py` 和 `tools/validate_pct.py` 用来找自己场景的拐点。

**阈值**
异常分**越大越差**（与 YOLO 的 score 方向相反）。阈值取"OK 侧最大值"与"NG 侧最小值"之间的
空档，见 `tools/calibrate_threshold.py`。正常料平均分记录在 `meta_v2.json` 的 `normal_mean`。

**骨干**
默认 `resnet50`。`tools/try_backbones.py` 做过 resnet18/34/50、wide_resnet50 的对比：
更小的骨干更快但小缺陷灵敏度下降，纹理类场景 wide_resnet50 更稳。

---

## 阈值标定

```bash
# 用「无偏 OK 分数 + NG 分数」联合扫描
python tools/calibrate_threshold.py

# 一体化 ONNX 模型的全量标定
python tools/calibrate_e2e.py
```

标定需要两批数据：**建库时没用过的 OK 图**、**若干 NG 图**。
只用建库的 OK 图算阈值会偏乐观（那些图已经被记忆库"记住"了），
应使用 `tools/validate_ok_batch.py` 的留出法。

---

## 性能参考

ONNX Runtime，输入按 ROI 裁剪后（单通道灰度），单张推理中位数：

| 场景 | CPU | GPU (RTX 3060 Laptop) |
|---|---|---|
| PatchCore（ROI 1236×1249） | 60.7 ms | 14.7 ms |

- GPU 侧建议设置 `gpu_mem_limit`（如 2GB）。默认 arena 会按倍数扩张，
  在 4GB 显存卡（GTX 1650 等）上容易被啃光。
- ORT 1.17.x 的 GPU 包锁定 **CUDA 11.8 + cuDNN 8**（依赖库里是 `cudart64_110` /
  `cublas64_11` / `cudnn64_8` 这些 ABI 后缀），装 CUDA 12.x 无效。

---

## 工具脚本（tools/）

开发过程中沉淀的分析脚本，按用途分组。多数脚本头部有 docstring 说明，
部分脚本里写死了本机路径，用前先改。

**建库 / 导出**
`train_custom.py` 用自建正常样本建库 · `train_own.py` 自建数据集建库 ·
`build_v2.py` 重建 v2 模型（ROI 外置） · `export_onnx.py` 导出 ONNX + 记忆库 ·
`export_e2e_onnx.py` 导出一体化 ONNX（图进、分数+热图出） ·
`export_fast_base.py` 导出提速骨干模板 · `rebuild_cam.py` 按当前骨干重建某相机库 ·
`repack_cam.py` 换骨干重拼 ONNX（库不动） · `custom_dataset.py` 自定义数据集适配器

**阈值标定**
`calibrate_threshold.py` · `calibrate_v2.py` · `calibrate_e2e.py`

**验证 / 评估**
`validate_ng.py` NG 检出能力 · `validate_ok_batch.py` OK 留出法（误报率） ·
`validate_pct.py` coreset 比例影响 · `eval_ng.py` / `eval_ng2.py` NG 评估 ·
`eval_all.py` MVTec AD 15 子类批量评测 · `verify_onnx.py` 与 PyTorch 数值一致性 ·
`verify_e2e.py` 一体化 ONNX 与原链路一致性 · `verify_scores.py` 新旧模型分数与耗时对比 ·
`check_resize_onnx.py` 下采样实现对比

**跨机台 / 归一化**
`crosscam_validate.py` 跨相机库通用性 · `test_cross_machine.py` 归一化能否消除分数偏移 ·
`test_norm_verdict.py` 归一化的能力边界（3 组实验） · `norm_pipeline.py` 跨机台图像归一化 ·
`site_config.py` 多机台部署配置

**性能基准**
`bench_backends.py` torch-GPU / torch-CPU / onnx-CPU 三后端对比 ·
`bench_cpu.py` CPU 推理速度 · `bench_knn_onnx.py` 记忆库烘焙进 ONNX 图 ·
`bench_tradeoff.py` 速度与精度权衡 · `try_backbones.py` 骨干对比

**分析 / 复核 / 报告**
`heatmap_123.py` / `heatmap_selfbuilt.py` 高分来源定位 · `vis_ng.py` NG 检出可视化 ·
`review_top.py` 可疑图复核 · `probe_test_folder.py` / `crossval_test_folder.py` 交叉验证 ·
`check_ng_corr.py` / `check_ng_overlap.py` 数据泄漏核查 ·
`make_report*.py` 各类 HTML 报告 · `check_env.py` 环境自检

---

## 常见问题

**Q：正常料被判成 NG**
先查三件事：① 建库用的 OK 图里有没有混入真实缺陷（`tools/make_report_audit.py` 可以审查）；
② 调用方裁剪/缩放方式是否与建库一致；③ 库规模是不是太小。

**Q：NG 检不出来**
看热图（`tools/heatmap_*.py`）确认缺陷位置有没有异常响应。若热图上毫无反应，
说明该缺陷在特征空间里和正常纹理不可分，换骨干或加大库规模试试；
若热图有响应但面积小，是 `min_area` 卡掉了。

**Q：换一台机器就全乱了**
正常现象，见上文"为什么每台相机必须独立建库"。不要试图用一个库跑多台机器。

---

## License

[Apache License 2.0](LICENSE)
