# 模型选型与下载

调研日期：2026-10-26。本项目没有下载模型，也不会在业务运行中自动拉取 ASR 权重。

## 默认组合

| 组件 | 本项目使用的模型 | 用途 |
| --- | --- | --- |
| ASR | `Qwen/Qwen3-ASR-1.7B` | 中文技术会议识别，支持中英文混合内容 |
| 强制对齐 | `Qwen/Qwen3-ForcedAligner-0.6B` | 给 ASR 文字补充字/词时间戳 |
| 可选声音分组 | `pyannote/speaker-diarization-community-1` | 在单场视频内区分声音来源，不识别真实姓名 |
| 纠错与纪要 | `.env` 指定的 LLM 服务 | 中文纠错、信息提炼、跨章节综合 |

Qwen 官方于 2026-01-29 发布 ASR 系列，包含 1.7B/0.6B，支持中文及多种方言；对齐模型支持包括中文在内的 11 种语言。2026-06-26 又提供了原生 Transformers 的 `-hf` 版本。本项目选择官方 `qwen-asr==0.0.6` 推理工具及原始模型目录，固定其配套 `transformers==4.57.6`。**不要直接把这里的模型替换成 `-hf` 变体**，那是另一套加载 API。[官方项目与发布时间](https://github.com/QwenLM/Qwen3-ASR)、[模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)。

这是兼顾中文识别、时间对齐和部署复杂度的工程选择，不是本项目实测后的精度排名。1.7B 作为默认质量档；资源紧张时可换成 `Qwen/Qwen3-ASR-0.6B` 并修改 `ASR_MODEL_PATH`。最终识别效果要在本次论坛音频上抽样评估。

## 方式一：ModelScope（国内网络优先）

安装环境后，在项目根目录执行：

```bash
.venv-server/bin/pip install -c constraints-gpu.txt modelscope
.venv-server/bin/modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
.venv-server/bin/modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/Qwen3-ForcedAligner-0.6B
```

小模型可选：

```bash
.venv-server/bin/modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./models/Qwen3-ASR-0.6B
```

上述下载途径来自 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)。下载命令本身可以访问网络；运行转写时则使用本地目录。

## 方式二：Hugging Face

```bash
.venv-server/bin/pip install 'huggingface_hub[cli]>=0.34,<1'
# .env 中的离线开关由业务程序读取，不影响直接执行的下载 CLI。
HF_HUB_OFFLINE=0 .venv-server/bin/hf download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
HF_HUB_OFFLINE=0 .venv-server/bin/hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B
```

无需下载模型时，不要执行上述命令。若服务器无法联网，可以在另一台机器下载整个模型目录后复制到 `models/`。不要只复制权重文件；配置、处理器、词表和相关辅助文件也必须保留。

默认配置：

```dotenv
ASR_MODEL_PATH=./models/Qwen3-ASR-1.7B
ALIGNER_MODEL_PATH=./models/Qwen3-ForcedAligner-0.6B
ASR_WORD_TIMESTAMPS=true
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

模型路径只接受本地目录。预检会检查配置和权重是否存在，加载仍可能发现下载不完整。先 `doctor`，再用短音频验收。

## 可选：声音分组

community-1 的官方流程要求先在 Hugging Face 接受模型使用条件，并使用有访问权限的个人 token 下载。它提供 exclusive speaker diarization，便于把声音标签与词时间戳匹配。[官方模型卡及离线使用](https://huggingface.co/pyannote/speaker-diarization-community-1)。

```bash
# 1. 在浏览器访问模型卡，登录并接受使用条件。
# 2. 安装可选依赖；固定 pyannote.audio 4.0.3 / torch 2.8.0。
bash scripts/setup.sh speakers

# 3. 交互登录，勿把 token 写进命令历史或提交到代码仓库。
.venv-server/bin/hf auth login

# 4. 下载整份仓库，保留所有子目录。
HF_HUB_OFFLINE=0 .venv-server/bin/hf download pyannote/speaker-diarization-community-1 --local-dir ./models/speaker-diarization-community-1
```

设置：

```dotenv
DIARIZATION=true
DIARIZATION_MODEL_PATH=./models/speaker-diarization-community-1
# 0 = 自动估计；已知人数时可填写正整数
DIARIZATION_NUM_SPEAKERS=0
PYANNOTE_METRICS_ENABLED=0
```

再次运行 `run`，ASR 缓存仍可复用，只新增声音分组和下游整理。模型按整场音频分析，会比默认模式使用更多 CPU 内存，且多人重叠、远场收音仍可能分错。无可靠词对齐时，会退回片段级主要说话人并提示，不保证短插话能正确划分。

Qwen 模型标示 Apache-2.0；community-1 标示 CC-BY-4.0，下载和再分发时请保留相应许可及署名信息，具体以模型仓库为准。
