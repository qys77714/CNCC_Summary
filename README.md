# CNCC 中文会议主题总结

把 video/ 中的中文技术会议视频转换为一篇**按明显主题详细展开的 Markdown 总结**。主题的数量、名称、顺序和内部写法由 LLM 根据实际内容决定，可使用段落、小标题、比较表、公式或简短示例。

正文以会议内容为主体，允许适量补充通用背景与原理解释。默认不做 LLM 逐字稿校正，不生成会议资料、导航、引用、待办、质量栏目或逐字稿附录。

## 已经跑过旧版：如何更新总结

保留服务器上每场的 transcript.raw.json，更新代码后在项目根目录执行：

~~~bash
.venv-server/bin/pip install -e .
.venv-server/bin/cncc-minutes summarize
~~~

此命令从**原始 ASR 转写**重新整理，自动忽略旧版摘要与纠错缓存，不加载 GPU，也不重新识别。旧 minutes.md 会保存为 minutes.previous.md。只处理指定一场时：

~~~bash
.venv-server/bin/cncc-minutes summarize --file 'video/某场会议.mp4'
~~~

新版不再生成校正稿、纠错记录或质量报告；旧输出文件会保留，但不会参与总结。CORRECTION_CHARS、INCLUDE_TRANSCRIPT 两项旧配置已经停用，可以从原有 .env 中删除。

如果原始转写本身有大段术语表复读、漏句或识别错误，重新总结无法恢复缺失的演讲。新版默认关闭 ASR 术语表注入；需要重新识别时执行 run，新的识别配置会使旧 ASR 缓存失效，已导出的音频仍可复用。

## 首次部署

目标环境：Linux、Python 3.11/3.12、4090 24GB 等 NVIDIA GPU。复制整个目录后：

~~~bash
# 使用单独的服务器环境，不复用从 Mac 复制的 .venv
bash scripts/setup.sh

# 下载模型；项目没有预装权重
.venv-server/bin/pip install -c constraints-gpu.txt modelscope
.venv-server/bin/modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
.venv-server/bin/modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./models/Qwen3-ForcedAligner-0.6B
~~~

编辑 .env，填写：

~~~dotenv
LLM_BASE_URL=https://你的服务地址/v1
LLM_API_KEY=你的密钥
LLM_MODEL=你的模型名
# 同一台机器四张卡
GPU_DEVICES=0,1,2,3
~~~

服务需兼容 /chat/completions，BASE_URL 包含服务要求的完整前缀，程序追加 /chat/completions。本地免鉴权服务可填 LLM_API_KEY=local。推荐至少 64K 上下文，并根据模型能力设置足够的输出上限。

~~~bash
.venv-server/bin/cncc-minutes doctor --llm
.venv-server/bin/cncc-minutes run
~~~

doctor --llm 会发一条小请求；run 和 summarize 会消耗所配置服务的额度。音视频不上传给 LLM，相关转写和内部材料会发送给配置的服务。

## 总结流程

1. ffmpeg 导出标准音频，在停顿附近切分，Qwen3-ASR 识别并可选对齐时间。
2. 直接读取原始转写，排除明显的术语表回声和相邻整段重复；不改写原始文件。
3. LLM 分批阅读，保留技术机制、论证、例子、实验条件、数字和局限的详细材料。
4. 通览材料目录，将跨时间窗口的同一主题归并，确定全篇的自然主题结构。
5. 逐主题撰写自由 Markdown。长主题分批续写并保留所有正文，不再对整篇做高度压缩。

内部阅读笔记与主题归属使用 JSON 检查材料编号，**这些字段不会规定最终 Markdown 的栏目**。最终正文由 LLM 直接撰写，程序保留表格、公式、代码块和段落格式。

## 写作偏好

~~~dotenv
# 允许适量通用背景解释；补充必须与会议观点区别开
SUMMARY_ALLOW_KNOWLEDGE=true

# 自然语言说明偏好，可留空，不需要修改代码
SUMMARY_STYLE=按明显主题充分展开，解释机制、案例与观点，不要写成几个简短要点。

# 一批原始转写的输入字符预算
SUMMARY_CHARS=10000
# 全场主题目录、每次写作材料的字符预算
REDUCE_INPUT_CHARS=16000

LLM_MAX_TOKENS=8192
~~~

补充知识用于帮助理解，不能编造报告者、论文出处、实测结果，也不能补齐录音缺失部分。关闭 SUMMARY_ALLOW_KNOWLEDGE 后，仅依据会议材料写作。背景补充不会被包装成报告者原话或会议结论。

输出被截断时，程序会明确失败；可增加 LLM_MAX_TOKENS 或减少输入预算。预算是字符数而非 token，此外还有提示词、schema 和输出。具体参数见 [部署文档](docs/DEPLOYMENT.md)。

## 并行和恢复

四卡同机：每张卡一个常驻识别进程，共享视频任务队列。识别完成的视频进入 LLM 整理，GPU 可继续识别下一场。LLM 请求共享并发和 RPM 限制。

四台单卡服务器：每台设置 GPU_DEVICES=0，分别运行：

~~~bash
# 四台分别使用 0、1、2、3；要求输入视频文件集合完全相同
.venv-server/bin/cncc-minutes run --shard-count 4 --shard-index 0
~~~

音频、ASR 片段、阅读笔记、主题规划和写作请求都有缓存。中断后重跑原命令即可；一场失败不影响其他场。部署细节见 [多机说明](docs/DEPLOYMENT.md)。

## 常用命令

~~~bash
# 只看任务分配
.venv-server/bin/cncc-minutes plan
# 只导出音频，无需 GPU
.venv-server/bin/cncc-minutes run --until extract
# 只识别，不调用 LLM
.venv-server/bin/cncc-minutes run --until transcribe
# 查看任务进度
.venv-server/bin/cncc-minutes status
# 汇集多台服务器结果后更新索引
.venv-server/bin/cncc-minutes index
# 指定配置文件，--env 放在子命令前
.venv-server/bin/cncc-minutes --env .env.server2 summarize
# 模拟输入和模拟 API，不需要模型或真实网络
.venv-server/bin/cncc-minutes demo
~~~

## 输出

| 文件 | 用途 |
| --- | --- |
| minutes.md | 主题式详细总结，仅保留正文 |
| summary.json | 主题规划、详细材料、各主题正文及输入过滤记录，用于恢复和调试 |
| transcript.raw.json / .md | 原始转写，保持独立，不附到总结中 |
| subtitles.raw.srt | 原始转写字幕 |
| audio.wav | 提取的标准音频 |
| cache/ | ASR 和 LLM 请求缓存 |
| status.json / events.jsonl | 进度、请求用量和阶段日志 |
| report_status.json | 本轮总结是否完成 |
| error.json | 失败时的错误信息 |
| output/README.md | 多场结果索引，独立于单场总结 |

原始时间戳和可选说话人标签仍可用于排查 ASR；真实姓名可在 speaker_names.json 中人工映射，但不强制加入最终正文。glossary.txt 默认用于识别明显词表回声；设置 ASR_USE_GLOSSARY=true 才会送给 ASR，须先验证低音量或静音处不会复读提示词。

本机已完成五场视频的音频导出；本轮改造通过 **35 项 CPU/模拟服务测试**。新版主题总结尚未使用你的真实 LLM 服务验收。示例 [minutes.md](examples/demo/simulated-meeting/minutes.md) 使用模拟输入与模拟响应，不对应真实视频。

更多文档：[模型下载](docs/MODELS.md) · [部署与排错](docs/DEPLOYMENT.md) · [架构](docs/ARCHITECTURE.md) · [设计依据](docs/RESEARCH.md) · [验证记录](docs/VALIDATION.md)。
