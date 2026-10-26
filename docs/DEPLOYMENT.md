# 部署、运行与排错

## 环境与模型

建议 Linux x86_64、Python 3.11/3.12。每张 GPU 一个识别进程，默认 BF16、SDPA、每批 2 个不超过 25 秒的片段。4090 24GB 是目标配置；当前没有在服务器测量峰值显存或吞吐。

建议主机至少 64GB RAM；四进程且开启整场声音分组时建议 128GB。依赖、模型、当前音频和运行缓存建议预留至少 40GB 磁盘。

~~~bash
bash scripts/setup.sh
~~~

脚本创建 .venv-server，安装 PyTorch/torchaudio 2.8.0 的 CUDA 12.8 wheel，再安装项目。核心约束在 constraints-gpu.txt，安装后的全部版本写入 environment.freeze.txt。服务器需要可运行这些 CUDA wheel 的驱动，建议较新的 570 系列或更新驱动并实际检查；默认不要求编译 FlashAttention。安装依据见 [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。

基础音频处理优先用系统 ffmpeg，缺少时使用 imageio-ffmpeg 分发的程序。可选 pyannote/torchcodec 建议安装系统 ffmpeg 及动态库：

~~~bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
~~~

模型下载见 [MODELS.md](MODELS.md)。运行时只加载本地 ASR/对齐/声音分组目录。

## 配置

相对路径以 .env 所在目录为根。系统环境变量优先；安装脚本不会覆盖已有 .env。不要公开密钥。

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| LLM_BASE_URL / LLM_API_KEY / LLM_MODEL | 占位 | 兼容服务地址前缀、密钥、模型名；本地免鉴权 key 可填 local |
| LLM_CONCURRENCY | 4 | 单次运行所有视频共享的最大请求并发 |
| LLM_REQUESTS_PER_MINUTE | 0 | 单次运行共享的发起请求速率；0 不限制 |
| LLM_TIMEOUT_SECONDS | 180 | 请求超时 |
| LLM_RETRIES | 4 | 包括首次在内的最多尝试次数 |
| LLM_MAX_TOKENS | 8192 | 单次输出 token 上限，需适合模型和主题写作长度 |
| LLM_RESPONSE_FORMAT | json_object | 只作用于内部阅读/规划；不支持时填 none。正文写作始终直接请求 Markdown |
| LLM_EXTRA_BODY | {} | 服务特有参数 JSON，不可覆盖核心字段 |
| SUMMARY_CHARS | 10000 | 一组原始转写的 JSON 字符预算 |
| REDUCE_INPUT_CHARS | 16000 | 主题目录和写作材料的字符预算，至少 4200 |
| SUMMARY_ALLOW_KNOWLEDGE | true | 允许适量通用知识解释，要求与会议观点区分 |
| SUMMARY_STYLE | 空 | 自然语言写作偏好，不预设各主题统一栏目 |
| MEETING_CONTEXT | 中文技术论坛 | 会议背景，不能混入没有来源的实测结论 |
| ASR_USE_GLOSSARY | false | 显式开启后才把 glossary.txt 注入 ASR；需防止词表回声 |

建议使用至少 64K 上下文、擅长中文长文与结构化输出的模型。字符预算不等于 token；提示、上下文、schema 和输出也占空间。LLM_MAX_TOKENS 是每次写作的上限，不是整篇总长度上限。

长主题可能多次续写。响应 finish_reason=length 会明确失败，已完成的其他请求保留；增加输出上限或降低输入预算后重跑。思考模型还需为思考预留 token，可按服务支持情况通过 LLM_EXTRA_BODY 关闭思考。

旧版 CORRECTION_CHARS、INCLUDE_TRANSCRIPT 不再读取；LLM 校正、固定摘要字段及逐字稿附录已经移除。现有 .env 中保留旧变量不会触发旧行为。

转写与内部材料会发送到所配置 LLM 服务，音视频不上传。完全离线需把 LLM 也部署在本地/内网；ASR 离线开关只约束本地模型。若 LLM 与 ASR 同机，GPU_DEVICES 应排除给 LLM 使用的显卡。

## 已有转写，直接更新总结

~~~bash
.venv-server/bin/pip install -e .
.venv-server/bin/cncc-minutes summarize
~~~

从每场 transcript.raw.json 开始，跳过 ASR 和旧纠错稿。改变写作偏好、知识补充开关或模型时重新运行 summarize，相关缓存自动失效。旧版 summary.json 格式不会被当成新结果。

新版不再生成校正逐字稿、纠错审计及质量报告；服务器上原有这些文件会保留作为历史产物。不要通过只改旧 minutes.md 来改变后续生成行为。

## 单机四卡

~~~dotenv
GPU_DEVICES=0,1,2,3
~~~

~~~bash
.venv-server/bin/cncc-minutes doctor --llm
.venv-server/bin/cncc-minutes run
~~~

每张卡动态领取完整视频，一个视频不跨卡拆分。某场识别完成即可进入 LLM 线程，GPU 接着识别下一场。

若外层 CUDA_VISIBLE_DEVICES=2,3，内部填 GPU_DEVICES=0,1，程序按外层分配映射到物理卡。不要再填物理编号 2,3。

## 四台单卡服务器

每台复制相同、固定的输入视频集合，设置 GPU_DEVICES=0 和独立输出根。四台分别执行：

~~~bash
# 服务器 0
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 0
# 服务器 1
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 1
# 服务器 2
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 2
# 服务器 3
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 3
~~~

分配按文件排序轮转，当前五场分为 2/1/1/1。可先用 plan --shard-count 4 --shard-index 0 查看分配。一批完成前不要只给部分机器增删文件，否则分配变化会重复或遗漏。

共享文件系统时使用不同的同层级输出根，例如 OUTPUT_DIR=./output-node0、./output-node1。同一输出根有独占锁。全部结束后合并各视频目录，保留原名，用 index 更新索引。

各台的 LLM 并发/RPM 独立计算，使用同一账户时要按总额度分配。

## 恢复与排错

- 中断后执行原命令：已导出的音频、已识别片段和成功的 LLM 请求会复用。
- 新运行开始时，旧 minutes.md 移到 minutes.previous.md；只有状态 complete 才代表本轮成功。
- 状态查看用 status；整批结果在 last_run.json，单场错误在 error.json。
- 429、网络错误、部分 5xx、结构或正文校验失败会有限重试；认证/参数类 4xx 立即失败。
- 源视频完整哈希可检测内容替换。模型指纹使用文件元数据和小配置内容；手工替换大权重应更新修改时间。
- 修改 ASR 设置或开启/关闭词表后执行 run；只更新总结执行 summarize。
- .env 只在命令启动时读取，不热加载。

| 问题 | 处理 |
| --- | --- |
| CUDA 不可用 | 检查驱动、CUDA PyTorch wheel、外层设备分配 |
| GPU 批次显存不足 | 程序二分当前批次；仍失败可设 batch=1、片长=15 或换 0.6B |
| 子进程意外退出 | 查看系统 OOM、主机内存、单场错误后重跑 |
| 词表复读 | 默认保持 ASR_USE_GLOSSARY=false；若旧转写大量污染，需要 run 重识别 |
| LLM 400 | 核对地址、模型、上下文长度、格式和额外参数 |
| JSON 模式不支持 | LLM_RESPONSE_FORMAT=none；内部阅读/规划仍需返回合法 JSON |
| 输出截断 | 增大 LLM_MAX_TOKENS，或降低 SUMMARY_CHARS / REDUCE_INPUT_CHARS |
| 主题分配校验反复失败 | 换更可靠的模型或降低输入目录预算，先用短样本验收 |
| 没有实质主题 | 检查音轨和原始转写，程序不会用知识补充编造缺失演讲 |

验收方法见 [VALIDATION.md](VALIDATION.md)。
