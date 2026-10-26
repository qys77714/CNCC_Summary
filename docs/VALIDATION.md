# 验证记录与服务器验收

## 本轮改造验证

日期：2026-10-26。本机为 macOS Apple Silicon / Python 3.12.14，没有下载或执行真实 GPU 模型。本轮没有调用真实 LLM 服务。

**35 项测试通过**，静态检查通过：

~~~bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
bash -n scripts/setup.sh
~~~

服务器环境对应使用 .venv-server/bin/python 和 .venv-server/bin/ruff。仅测试 CPU/模拟流程可执行 bash scripts/setup.sh cpu。

主要覆盖：

- 真实 ffmpeg 转换、精确采样区间、音频缓存、多进程失败隔离。
- ASR API 参数、GPU 外层分配、片段恢复、OOM 二分后顺序保持。
- 默认 ASR 上下文不注入术语表。
- 新版跳过纠错，不读取旧校正稿/旧结构化摘要，不修改原始转写。
- 跨多个阅读窗口的同主题归并；目录较大时分层规划，不遗漏或重复分配材料。
- 长主题多批续写保留开头和末尾的具体实验条件，不再做最终压缩。
- JSON 内部校验、Markdown 正文请求不携带 JSON response_format。
- 表格、公式、代码格式保留；旧栏目及原文引用不进入正文。
- 姓名映射、知识补充开关、写作风格传入请求。
- 重试、缓存、重建失败、无实质内容时明确失败。

模拟服务用于验证控制逻辑，不能证明真实模型的主题划分或文字质量。示例 examples/demo/simulated-meeting/minutes.md 明确采用模拟输入与模拟响应。

## 用户提供的旧纪要检查

检查了 /Users/qys/Downloads/minutes.md。其附录含 824 段转写，保守词表回声规则标记 96 段，规则未修改附件。该计数只说明这些文字与配置词表高度吻合，不是声学检测，也不证明其余段落全部有效。

总结输入会排除这些明显回声，其余闲聊和无实质内容由阅读提示处理。旧结果中缺失的演讲不能通过新总结补全；应回查原始音轨并重跑 ASR。

## 历史音频验证

此前对五个真实视频完成并行音频导出，均为 16kHz 单声道 PCM WAV，音轨合计约 17:46:12，2,047,123,042 字节。此轮保留这些结果，没有重复导出视频或进行 GPU 识别。

## 服务器验收

已有完整转写时，先选择一场运行：

~~~bash
.venv-server/bin/cncc-minutes summarize --file 'video/某场会议.mp4'
~~~

检查：

1. 是否形成几个清晰的内容主题，而非按时间切片或每段重复固定栏目。
2. 同一主题跨时间出现的材料是否合并，方法、例子、实验条件是否充分保留。
3. 每个主题内部结构是否适合其内容，而非统一套“背景/方法/结论”模板。
4. 背景补充能否与会议陈述区分，有无臆造数据、论文或人名。
5. 正文是否没有资料、导航、引用、待办、附录、质量清单。
6. 再运行相同命令，确认缓存复用；修改 SUMMARY_STYLE 后确认重建生效。

如果原始转写明显污染，准备短音频进行 GPU 验收，再处理整场：

~~~bash
mkdir -p samples
ffmpeg -ss 00:10:00 -i 'video/某场会议.mp4' -t 120 -vn -ac 1 -ar 16000 samples/smoke.wav
.venv-server/bin/cncc-minutes doctor --llm
GPU_DEVICES=0 .venv-server/bin/cncc-minutes run --file samples/smoke.wav
~~~

需在服务器实测模型加载、显存/速度、长时稳定性、可选声音聚类、真实 LLM 格式兼容和内容质量。先验收短样本，再运行全部视频。
