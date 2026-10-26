import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .media import ffmpeg_binary
from .pipeline import discover, run
from .storage import read_json


def doctor(cfg, cpu=False, check_llm=False):
    failures = []
    print(f"CNCC Minutes {__version__} | Python {sys.version.split()[0]}")
    try:
        binary = ffmpeg_binary(cfg.ffmpeg_path)
        result = subprocess.run([binary, "-version"], capture_output=True, text=True, check=True)
        print("音频工具：" + result.stdout.splitlines()[0])
    except Exception as e:
        failures.append(f"音频工具不可用：{e}")
    files = discover(cfg.input_dir)
    print(f"输入：{cfg.input_dir}，发现 {len(files)} 个媒体文件")
    print(f"输出：{cfg.output_dir}")
    if not cpu:
        try:
            import torch
            print(f"PyTorch {torch.__version__} | CUDA runtime {torch.version.cuda}")
            if not torch.cuda.is_available():
                failures.append("CUDA 不可用")
            for device in cfg.gpus:
                i = int(device)
                if i >= torch.cuda.device_count():
                    failures.append(f"GPU {i} 不可见")
                else:
                    info = torch.cuda.get_device_properties(i)
                    print(f"GPU {i}: {info.name}, {info.total_memory / 1024**3:.1f} GiB")
            importlib.metadata.version("qwen-asr")
        except Exception as e:
            failures.append(f"GPU 依赖不完整：{e}")
        from .asr import require_model
        for model in [cfg.asr_model] + ([cfg.aligner_model] if cfg.word_timestamps else []):
            try:
                require_model(model)
                print(f"本地模型：{model}")
            except ValueError as e:
                failures.append(str(e))
        if cfg.diarization:
            try:
                importlib.metadata.version("pyannote.audio")
                if not (cfg.diarization_model / "config.yaml").is_file():
                    raise ValueError("缺少 community-1 config.yaml")
            except Exception as e:
                failures.append(f"说话人依赖不完整：{e}")
    if check_llm:
        from .llm import LLMClient
        try:
            cfg.validate_llm()
            from pydantic import BaseModel

            class Ping(BaseModel):
                ok: bool

            client = LLMClient(cfg)
            try:
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    result = client.structured(Path(temp), "doctor", '返回 JSON {"ok": true}', {}, Ping)
                    if not result.ok:
                        failures.append("LLM 连通测试返回 false")
                print(f"LLM 连通：{cfg.llm_model}")
            finally:
                client.close()
        except Exception as e:
            failures.append(f"LLM 检查失败：{e}")
    if failures:
        print("\n".join(f"失败：{message}" for message in failures))
        return 1
    print("检查通过" + ("（仅 CPU 工具；未验证 GPU、模型或 LLM）" if cpu and not check_llm else ""))
    return 0


def main():
    parser = argparse.ArgumentParser(description="中文长会议视频 → ASR → 纠错 → 可追溯 Markdown 纪要")
    parser.add_argument("--env", type=Path, default=Path(".env"), help=".env 路径，相对配置以其所在目录为根")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("run", "运行完整流程"), ("summarize", "复用已有转写，仅运行 LLM 及报告"),
                            ("plan", "只列出本机要处理的媒体文件，不运行模型")]:
        sub = subs.add_parser(name, help=help_text)
        sub.add_argument("--file", type=Path, action="append", default=[], help="仅处理此文件，可重复；默认扫描输入目录")
        sub.add_argument("--shard-index", type=int, default=0, help="本机分片编号，从 0 开始")
        sub.add_argument("--shard-count", type=int, default=1, help="服务器总数")
        if name == "run":
            sub.add_argument("--until", choices=["extract", "transcribe", "complete"], default="complete")
    sub = subs.add_parser("doctor", help="检查依赖、GPU、本地模型；不下载模型")
    sub.add_argument("--cpu", action="store_true", help="只检查本地 CPU 工具")
    sub.add_argument("--llm", action="store_true", help="额外发送一条小请求验证 LLM 配置，会消耗服务额度")
    subs.add_parser("status", help="查看现有任务进度")
    subs.add_parser("index", help="汇集结果后重新生成 Markdown 索引")
    subs.add_parser("demo", help="用明确标注的模拟资料生成示例，不识别真实视频、不调用网络")
    args = parser.parse_args()
    try:
        cfg = Settings.load(args.env)
        if args.command in {"run", "summarize", "plan"}:
            videos = discover(cfg.input_dir, args.file, args.shard_index, args.shard_count)
            print(f"本次处理 {len(videos)} 个文件；GPU 配置 {','.join(cfg.gpus)}")
            if args.command == "plan":
                print("\n".join(str(p) for p in videos))
                code = 0
            elif not videos and args.shard_count > 1:
                print("当前分片没有任务。")
                code = 0
            else:
                code = run(cfg, videos, getattr(args, "until", "complete"), llm_only=args.command == "summarize")
        elif args.command == "doctor":
            code = doctor(cfg, args.cpu, args.llm)
        elif args.command == "status":
            statuses = sorted(cfg.output_dir.glob("*/status.json"))
            for path in statuses:
                data = read_json(path)
                details = {k: v for k, v in data.items() if k in {"completed", "total", "chapter", "batch", "level"}}
                print(f"{path.parent.name}\n  {data['time']} {data['stage']}：{data['message']} "
                      + json.dumps(details, ensure_ascii=False))
            if not statuses:
                print("暂无任务")
            code = 0
        elif args.command == "index":
            from .report import render_index
            from .storage import output_lock
            with output_lock(cfg.output_dir):
                render_index(cfg.output_dir)
            print(f"索引已更新：{cfg.output_dir / 'README.md'}")
            code = 0
        else:
            from .demo import generate_demo
            result = generate_demo(cfg)
            print(f"模拟示例已生成（不是视频识别结果）：{result}")
            code = 0
    except KeyboardInterrupt:
        print("\n已中断；再次执行原命令可复用已完成片段。", file=sys.stderr)
        code = 130
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
