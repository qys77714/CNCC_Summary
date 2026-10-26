# 部署、运行与排错

## 环境

建议使用 Linux x86_64、Python 3.11 或 3.12。每个 GPU 运行一个独立 ASR 工作进程，默认 BF16、SDPA、每批 2 个不超过 25 秒的片段。4090 24GB 是目标配置，实际峰值显存尚未在服务器上测量。建议主机至少 64GB RAM；四进程且开启整场声音分组时，建议 128GB。安装依赖、权重及保留当前素材音频，建议预留至少 40GB 可用磁盘。

```bash
bash scripts/setup.sh
```

脚本建立 `.venv-server/`，安装 PyTorch 2.8.0 / torchaudio 2.8.0 的 CUDA 12.8 wheel，然后安装本项目和官方 Qwen 推理依赖。核心组合固定在 `constraints-gpu.txt`；安装后的全部版本会写入 `environment.freeze.txt`，便于服务器验收后复现。它不是跨平台完整锁文件。

安装方式参照 [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。服务器应安装能运行 CUDA 12.8 wheel 的 NVIDIA 驱动，建议使用较新的 570 系列或更新驱动并通过实际 CUDA 检查；`nvidia-smi` 上的 CUDA 标识不等于 Python 实际使用的 runtime。旧驱动的兼容能力有条件，见 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。SDPA 模式不要求本机编译 FlashAttention 或安装完整 CUDA Toolkit。

基础转写使用系统 ffmpeg，缺失时使用随 `imageio-ffmpeg` 分发的可执行文件。可选 pyannote/torchcodec 建议同时安装系统 ffmpeg 及其动态库：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

不具备管理员权限时，可由管理员安装，或者使用环境中的 ffmpeg 并通过 `FFMPEG_PATH` 指定路径。

## LLM 配置

根目录已有 `.env` 占位配置，安装脚本不会覆盖它。相对路径均以 `.env` 所在目录为根；系统环境变量优先于 `.env`。不要把真实密钥写进命令参数或纳入版本控制。

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | 占位 | 完整服务前缀，如 `https://host/v1`，不包含末尾 `/chat/completions` |
| `LLM_API_KEY` / `LLM_MODEL` | 占位 | 服务密钥和真实模型名；本地免鉴权服务可用 `local` |
| `LLM_CONCURRENCY` | 4 | 多视频共享的最大并发请求数 |
| `LLM_REQUESTS_PER_MINUTE` | 0 | 单次运行共享的发起请求速率，0 不限制；多机时每台单独计算 |
| `LLM_TIMEOUT_SECONDS` | 180 | 单次请求超时 |
| `LLM_RETRIES` | 4 | 每次请求最多尝试次数，包括第一次 |
| `LLM_MAX_TOKENS` | 8192 | 最大输出 token；思考模型可能需要增大或关闭思考 |
| `LLM_RESPONSE_FORMAT` | `json_object` | 服务不支持该字段时设 `none`，仍要求正文能解析为 JSON |
| `LLM_EXTRA_BODY` | `{}` | 额外请求参数 JSON；如服务支持，可用 `{"enable_thinking":false}` |
| `CORRECTION_CHARS` | 4500 | 每组纠错 target 的 JSON 字符预算，另带少量前后文 |
| `SUMMARY_CHARS` | 10000 | 局部章节输入的 JSON 字符预算 |
| `REDUCE_INPUT_CHARS` | 16000 | 跨章节合并输入预算，包含原始片段引用，至少 4200 |
| `MEETING_CONTEXT` | 技术论坛说明 | 会议背景，可补充实际议程；不要混入无来源结论 |
| `INCLUDE_TRANSCRIPT` | true | 最终纪要附完整校正逐字稿；关闭后引用跳转外部 `transcript.md` |

建议选上下文至少 64K、擅长中文与结构化输出的模型。这里的预算是字符，不是 token，此外还有系统提示、术语表、JSON schema 和输出。不同模型分词不同；如果服务报上下文超长，降低三个字符预算、缩短术语表，或改用更大上下文模型。`LLM_EXTRA_BODY` 支持服务商特有参数，不允许覆盖模型、消息、输出上限等核心字段。

本地部署的任意兼容服务也可以使用。例如将已部署的 LLM 服务地址填为 `http://127.0.0.1:8000/v1`。本项目不负责安装/启动 LLM；若与 ASR 共用显卡，应在 `GPU_DEVICES` 中排除该卡，避免显存争用。

启用完整流程时，**转写文字、术语和背景会发送给你配置的 LLM 服务**，不会把视频或音频上传给 LLM。ASR 和可选声音分组在本地执行。需要完全离线时，把 LLM 地址设为内网本地服务；只有 ASR 离线开关不足以让整个流程离线。

## 单台机器四张卡

```dotenv
GPU_DEVICES=0,1,2,3
```

```bash
.venv-server/bin/cncc-minutes doctor --llm
.venv-server/bin/cncc-minutes run
```

主进程建立共享任务队列，四个常驻 GPU 进程动态领取视频。一个视频在一张卡上顺序分批识别，不跨卡拆同一视频。某场 ASR 结束后即转交 LLM 线程，GPU 可以继续识别下一场；LLM 和识别阶段重叠执行。

如果外层已设置 `CUDA_VISIBLE_DEVICES=2,3`，则 `.env` 的 `GPU_DEVICES` 应填写可见逻辑编号 `0,1`，程序会正确映射回外层分配的设备。不要填写 `2,3`。MIG/UUID 形式的外层设备分配可以保留，项目内部仍填数字逻辑编号。

## 四台机器，每台一张卡

每台都复制完整项目和**相同且固定的视频文件集合**，配置本机 `GPU_DEVICES=0`、模型路径及 LLM。各服务器的输出目录应互相独立。

```bash
# 每台先查看自己将处理哪些文件
.venv-server/bin/cncc-minutes plan --shard-count 4 --shard-index 0

# 服务器 0
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 0
# 服务器 1
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 1
# 服务器 2
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 2
# 服务器 3
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 3
```

分配方式为排序后按编号轮转。当前五个视频会分成 2/1/1/1，确保四台都有任务。在这一批完成前，不要只给部分机器增删输入，否则分配会变化。每台都须使用同样的 `--shard-count` 和不同的 `--shard-index`。这是无需协调服务的静态分片；跨机器没有全局动态负载均衡。

若机器共享同一文件系统，可用环境变量设置同层级的不同输出根，例如服务器 0 用 `OUTPUT_DIR=./output-node0`，服务器 1 用 `OUTPUT_DIR=./output-node1`。同一个输出根有独占锁，不允许同时启动两次运行。

全部完成后，把各服务器 `output/` 内的各视频目录合并到一个 `output/`，保留目录原名。不要在任务运行中移动缓存。可通过 `cncc-minutes index` 重新生成索引。相同视频集合和相同输入根层级会生成相同任务目录名；原文件内容变动会使缓存重新计算。

多台机器如果共用同一 LLM 账户，总并发和 RPM 是各台之和。例如总限制 60 RPM 时，每台可设 `LLM_REQUESTS_PER_MINUTE=15`。

## 恢复、更新与阶段运行

- 中断后直接重跑相同命令。音频、分片、已完成 ASR 片段和通过校验的 LLM 响应均可复用。
- API 超时、429、部分 5xx、非法 JSON、片段 ID 不匹配等会有有限重试。认证、地址、参数等 4xx 会立即失败并提示修改配置。
- 一场失败不会阻止其他场完成。`output/last_run.json` 记录整批成功/失败情况。
- 换 LLM、修改背景或说话人姓名：运行 `summarize`，不加载 GPU 模型。只修改姓名不会重新跑纠错，但章节与综合摘要会更新。
- 修改热词表后，`run` 会重新跑 ASR 及下游；若只希望对已有文字纠错，可用 `summarize`。
- 修改 ASR 模型、语言、时间戳、分片参数：运行 `run`，相关缓存自动失效。修改 batch size 不强制废弃已识别片段。
- `.env` 改变后必须启动新的命令；运行中的进程不会热加载。
- 不要手改 `transcript.corrected.json` 期待自动保留，重建可能覆盖。原始转写 JSON 是 `summarize` 的输入，手动修正其 text 后可重新总结；保留 id 和时间字段，并另行备份。
- 模型指纹使用文件元数据及小配置内容，不逐个完整哈希大权重；手动替换权重时应更新文件修改时间，或删除该视频的 ASR 缓存重新识别。
- 每次新运行开始时，旧 `minutes.md` 会移为 `minutes.previous.md`。只有 `report_status.json` 为 complete、索引状态为 complete 的纪要属于本轮成功结果。

## 排错

| 现象 | 处理 |
| --- | --- |
| `CUDA 不可用` | 检查 `nvidia-smi`、PyTorch 是否为 CUDA wheel、外层可见卡及驱动 |
| 多卡进程退出 | 查看视频目录 `error.json` 和系统 OOM 日志；减小 GPU 数或增加主机 RAM |
| 批次显存不足 | 程序会自动二分当前批次；仍失败时改 `ASR_BATCH_SIZE=1`、`ASR_CHUNK_SECONDS=15` 后重跑 |
| 单片显存不足 | 减小片长，或者改用 0.6B 模型；根据需要关闭词时间戳或声音分组 |
| 本地模型缺失 | 下载完整原始模型目录，不要混用 `-hf` 模型，核对 `.env` 相对路径 |
| LLM 400 | 核对模型、上下文长度、`response_format` 和额外参数；服务不支持格式字段时设为 none |
| LLM 401/403 | 替换 API key / 权限，检查服务前缀 |
| finish_reason=length | 增大最大输出 token、减少输入预算；思考模型可按服务能力关闭思考 |
| 引用/JSON 多次校验失败 | 换结构化输出能力更强的模型，先跑短样本验证；错误不会被当作成功纪要 |
| 对齐文字不一致 | 自动保留完整 ASR 原文，退回音频片段时间，并在质量报告标记 |
| 人名或专有词不准 | 补充术语表并回听；LLM 不能恢复 ASR 完全遗漏的信息 |
| 字幕一条很长 | 字幕按转写片段生成，适用于检索回放；没有实现影视字幕的字符数/阅读速度排版 |

先用短样本验收，再放开全量处理，步骤见 [VALIDATION.md](VALIDATION.md)。
