# CNCC 中文会议视频纪要

把 `video/` 中的中文会议视频转换为可回查原文的 Markdown 会议纪要。默认使用 **Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B** 在服务器 GPU 上识别，再通过你配置的 LLM 服务进行保守纠错、章节提炼和全场综合。

支持四卡并行、单卡运行、不同服务器分片、音频片段级断点续跑。每场生成一个自包含的 `minutes.md`，包含综合概要、技术要点、决议、明确待办、实际问答、术语、时间章节、质量提示和完整校正逐字稿。没有明确决议/待办时，会直接说明未提取。

## 在服务器上快速开始

目标环境：Linux、Python 3.11/3.12、NVIDIA GPU。复制整个项目后，在项目根目录执行：

```bash
# 1. 建立服务器独立环境，不复用从 Mac 复制来的 .venv
bash scripts/setup.sh

# 2. 下载两个模型。这里只提供命令，项目没有预装模型。
.venv-server/bin/pip install -c constraints-gpu.txt modelscope
.venv-server/bin/modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
.venv-server/bin/modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/Qwen3-ForcedAligner-0.6B
```

编辑根目录已有的 `.env`，至少替换：

```dotenv
LLM_BASE_URL=https://你的服务商地址/v1
LLM_API_KEY=你的密钥
LLM_MODEL=你的模型名
# 四张卡在同一台机器时
GPU_DEVICES=0,1,2,3
# 每台服务器只有一张卡时改为 0，并按部署文档使用 --shard-index
```

LLM 服务需兼容 `/chat/completions`，建议使用支持至少 64K 上下文、中文能力较强的指令模型。`BASE_URL` 要包含服务要求的完整前缀，程序会追加 `/chat/completions`。本地免鉴权服务的 API key 可填 `local`。配置详解见 [部署文档](docs/DEPLOYMENT.md)。

```bash
# 3. 检查 GPU、本地模型、LLM 连接（--llm 会发送一条小型请求）
.venv-server/bin/cncc-minutes doctor --llm

# 4. 并行处理全部视频
.venv-server/bin/cncc-minutes run

# 5. 在另一个终端查看进度
.venv-server/bin/cncc-minutes status
```

打开 `output/README.md` 查看结果索引，每个视频对应独立输出目录。全部成功时退出码为 0，部分失败为 1，手动中断为 130。失败不会阻止其他视频继续；修复后直接重跑原命令，已完成的识别片段和 LLM 请求会复用。

## 常用操作

```bash
# 查看任务分配，不调用模型
.venv-server/bin/cncc-minutes plan

# 只处理一场，文件名有中文或空格时要加引号
.venv-server/bin/cncc-minutes run --file 'video/某场会议.mp4'

# 不需要 GPU：仅导出音频
.venv-server/bin/cncc-minutes run --until extract

# 先完成本地转写，暂不配置/调用 LLM
.venv-server/bin/cncc-minutes run --until transcribe

# 已有转写时仅重新纠错/总结，不加载 GPU 模型
.venv-server/bin/cncc-minutes summarize

# 使用另一份配置，注意 --env 放在子命令前
.venv-server/bin/cncc-minutes --env .env.server2 run

# 不需要模型、GPU、真实 API：查看明确标注的模拟纪要
.venv-server/bin/cncc-minutes demo
```

`summarize` 不重新识别音频；调整 ASR 参数、术语热词或说话人模型后，应执行 `run`。只修改 LLM、纪要背景或人工说话人姓名时，可执行 `summarize`。

## 输出内容

| 文件 | 用途 |
| --- | --- |
| `minutes.md` | 最终完整纪要，默认附完整校正逐字稿；可独立复制阅读 |
| `audio.wav` | 16kHz、单声道、16-bit PCM 音频 |
| `transcript.raw.json` / `.md` | 带相对时间和片段 ID 的原始 ASR 转写 |
| `transcript.corrected.json` / `transcript.md` | 校正逐字稿，保留原文及待核对标记 |
| `subtitles.raw.srt` / `subtitles.srt` | 原始/校正字幕，按转写片段显示 |
| `corrections.json` | 原文、建议改法、实际采用文字和拒绝原因 |
| `summary.json` | 全场综合纪要和所有时间章节的结构化数据 |
| `quality.json` | 疑点、对齐警告、拒绝大幅改写的计数 |
| `status.json` / `events.jsonl` | 当前状态与阶段日志，含 API 返回的 token 用量 |
| `error.json` | 失败原因与调试堆栈，失败时生成 |
| `report_status.json` | 本轮纪要是否完成；旧纪要移至 `minutes.previous.md` |
| `cache/` | ASR 片段和 LLM 请求缓存，保留可节约重跑开销 |

默认报告中的时间引用跳转本文件的逐字稿锚点。逐字稿中的时间链接还会指向原视频的媒体时间片段；能否直接跳到视频位置取决于 Markdown 阅读器。SRT 的时间来自 ASR 对齐，纠错后的词没有重新对齐。

## 说话人与技术名词

`glossary.txt` 可填模型名、论文名、人名及常见错误写法；它同时用于 ASR 上下文提示和 LLM 纠错，不能保证专名一定正确。

默认关闭声音分组，转写标记为“未区分”。需要多人发言区分时，按 [模型文档](docs/MODELS.md) 下载可选的 pyannote community-1，再执行 `bash scripts/setup.sh speakers` 并设置 `DIARIZATION=true`。它按整场声音聚类，输出 `SPEAKER_00` 等编号；不会自动确认真实身份。人工确认后，可编辑 `speaker_names.json`：

```json
{
  "某场会议.mp4": {
    "SPEAKER_00": "主持人（人工确认）",
    "SPEAKER_01": "张老师（人工确认）"
  }
}
```

保存后运行 `summarize`，报告和摘要会更新姓名。各视频的编号彼此独立。

## 当前项目的验证状态

2026-10-26 已对当前 5 个真实视频完成并行音频导出，音轨合计约 **17:46:12**，WAV 共 **2.05 GB（十进制）**。音频已在 `output/`，整体复制至服务器后可复用，删除后也可重新导出。

已完成本机 CPU 自动化测试与模拟 LLM 流程，详见 [验证记录](docs/VALIDATION.md)。**没有下载任何 ASR/说话人模型，没有执行真实 GPU 推理，也没有用真实 LLM 生成这五场会议的纪要。** 示例目录中的纪要使用模拟资料，并有清晰标注。

## 更多文档

- [服务器环境、四卡/多机运行、恢复与排错](docs/DEPLOYMENT.md)
- [模型选型、下载与离线部署](docs/MODELS.md)
- [飞书能力调研与本项目设计](docs/RESEARCH.md)
- [处理流程、缓存与扩展](docs/ARCHITECTURE.md)
- [验证记录及服务器验收方法](docs/VALIDATION.md)

飞书式结构参考其官方的 [智能纪要说明](https://www.feishu.cn/hc/zh-CN/articles/244959839578)；本项目实现音频转写和离线纪要文件，不接入飞书账号。ASR 接口依据 [Qwen3-ASR 官方项目](https://github.com/QwenLM/Qwen3-ASR)。
