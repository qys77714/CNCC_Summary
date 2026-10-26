import multiprocessing as mp
import os
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .asr import ASREngine, build_segments
from .llm import LLMClient
from .media import MEDIA_EXTENSIONS, prepare_audio, prepare_chunks
from .report import mark_stale, render_index, srt_text, transcript_markdown
from .storage import atomic_text, digest, event, job_id, output_lock, read_json, source_signature, write_json
from .summarize import summarize_video


def discover(input_dir: Path, files=(), shard_index=0, shard_count=1):
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("分片要求 0 <= shard-index < shard-count")
    candidates = [Path(p).resolve() for p in files] if files else sorted(input_dir.rglob("*"))
    videos = sorted({p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS})
    # Balanced assignment for a fixed input snapshot. All servers must use the same input file set.
    return [p for i, p in enumerate(videos) if i % shard_count == shard_index]


def process_media(source, job, cfg, engine, until):
    event(job, "hash", "校验源视频内容")
    signature = source_signature(source)
    meta = prepare_audio(source, job, signature, cfg)
    if until == "extract":
        event(job, "extracted", "音频导出完成")
        return
    plan = prepare_chunks(job, meta, cfg)
    asr = engine.transcribe(job, plan)
    speakers = engine.diarize(job, meta)
    segments = build_segments(asr, speakers)
    raw_doc = {"source": str(source), "key": digest({"asr": asr["key"], "speakers": speakers["key"], "segments": 2}),
               "model": asr["model"], "segments": segments}
    write_json(job / "transcript.raw.json", raw_doc)
    names = cfg.speaker_names.get(source.name, {})
    atomic_text(job / "transcript.raw.md", "# 原始 ASR 转写\n\n" + transcript_markdown(segments, names))
    atomic_text(job / "subtitles.raw.srt", srt_text(segments))
    event(job, "transcribed", "转写完成", segments=len(segments))


def gpu_worker(device, cfg, tasks, results, until):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [x.strip() for x in visible.split(",")] if visible else None
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(device)] if devices else device
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    engine = ASREngine(cfg)
    while (task := tasks.get()) is not None:
        source, job = map(Path, task)
        results.put({"type": "started", "source": str(source), "job": str(job), "pid": os.getpid()})
        try:
            process_media(source, job, cfg, engine, until)
            results.put({"type": "done", "source": str(source), "job": str(job), "ok": True})
        except Exception as e:
            event(job, "failed", str(e))
            write_json(job / "error.json", {"error": str(e), "traceback": traceback.format_exc()})
            results.put({"type": "done", "source": str(source), "job": str(job), "ok": False, "error": str(e)})


def run(cfg, videos, until="complete", llm_only=False):
    if not videos:
        raise ValueError("没有找到匹配的视频或音频")
    if until == "complete":
        cfg.validate_llm()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not llm_only and visible and max(map(int, cfg.gpus)) >= len(visible.split(",")):
        raise ValueError("GPU_DEVICES 超出外层 CUDA_VISIBLE_DEVICES 的可见逻辑编号")
    jobs = [(source, cfg.output_dir / job_id(source, cfg.input_dir)) for source in videos]
    if len({job for _, job in jobs}) != len(jobs):
        raise ValueError("输入文件生成了重复任务目录；请把外部同名文件放进 INPUT_DIR 的不同子目录再运行。")
    failures = []
    interrupted = False
    with output_lock(cfg.output_dir):
        for source, job in jobs:
            job.mkdir(parents=True, exist_ok=True)
            mark_stale(job)
            event(job, "queued", "任务已排队", source=source.name)
        client = LLMClient(cfg) if until == "complete" else None
        pool = ThreadPoolExecutor(max_workers=cfg.llm_concurrency)
        llm_futures = {}
        processes = []

        def submit_llm(source, job):
            llm_futures[pool.submit(summarize_video, job, cfg, client)] = (source, job)

        try:
            if llm_only:
                for source, job in jobs:
                    try:
                        if not (job / "transcript.raw.json").exists():
                            raise ValueError("缺少 transcript.raw.json，请先运行转写")
                        if read_json(job / "audio.json")["key"] != digest({
                            "source": source_signature(source), "format": "pcm_s16le-mono-16000-v1"
                        }):
                            raise ValueError("源视频发生变化，请重新运行转写，不能复用旧逐字稿")
                        submit_llm(source, job)
                    except Exception as e:
                        failures.append({"source": str(source), "error": str(e)})
                        event(job, "failed", str(e))
            else:
                context = mp.get_context("spawn")
                tasks, results = context.Queue(), context.Queue()
                for source, job in jobs:
                    tasks.put((str(source), str(job)))
                for device in cfg.gpus[:len(jobs)]:
                    tasks.put(None)
                    p = context.Process(target=gpu_worker, args=(device, cfg, tasks, results, until))
                    p.start()
                    processes.append(p)
                completed, last_print = set(), 0.0
                while len(completed) < len(jobs):
                    try:
                        result = results.get(timeout=1)
                    except queue.Empty:
                        if not any(p.is_alive() for p in processes):
                            try:
                                result = results.get(timeout=1)
                            except queue.Empty:
                                break
                        else:
                            if time.monotonic() - last_print >= 15:
                                for _, job in jobs:
                                    if str(job) not in completed and (job / "status.json").exists():
                                        status = read_json(job / "status.json")
                                        print(f"[{job.name[:24]}] {status['stage']}: {status['message']}", flush=True)
                                last_print = time.monotonic()
                            continue
                    if result["type"] == "started":
                        print(f"开始处理：{Path(result['source']).name}", flush=True)
                        continue
                    completed.add(result["job"])
                    source, job = Path(result["source"]), Path(result["job"])
                    if result["ok"] and until == "complete":
                        submit_llm(source, job)
                    elif not result["ok"]:
                        failures.append({"source": str(source), "error": result["error"]})
                        print(f"处理失败：{source.name}：{result['error']}", flush=True)
                    else:
                        print(f"完成：{source.name}（{until}）", flush=True)
                for source, job in jobs:
                    if str(job) not in completed:
                        message = "GPU 工作进程意外退出，请检查系统 OOM / 驱动后原命令重跑"
                        failures.append({"source": str(source), "error": message})
                        event(job, "failed", message)
            for future in as_completed(llm_futures):
                source, job = llm_futures[future]
                try:
                    future.result()
                    print(f"纪要完成：{job / 'minutes.md'}", flush=True)
                except Exception as e:
                    failures.append({"source": str(source), "error": str(e)})
                    write_json(job / "report_status.json", {"status": "failed", "error": str(e)})
                    write_json(job / "error.json", {"error": str(e), "traceback": traceback.format_exc()})
                    event(job, "failed", str(e))
                    print(f"纪要失败：{source.name}：{e}", flush=True)
        except BaseException:
            interrupted = True
            raise
        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=10)
            pool.shutdown(wait=True, cancel_futures=True)
            if client:
                client.close()
            success_stage = {"complete": "complete", "extract": "extracted", "transcribe": "transcribed"}[until]
            succeeded = 0
            for source, job in jobs:
                status = read_json(job / "status.json")
                if status["stage"] == success_stage:
                    succeeded += 1
                elif not any(f["source"] == str(source) for f in failures):
                    message = "任务已中断，请原命令重跑" if interrupted else "任务未完成，请查看日志后重跑"
                    failures.append({"source": str(source), "error": message})
                    event(job, "interrupted" if interrupted else "failed", message)
            render_index(cfg.output_dir)
            write_json(cfg.output_dir / "last_run.json", {
                "total": len(jobs), "succeeded": succeeded, "failures": failures,
                "until": until, "llm_only": llm_only, "interrupted": interrupted,
            })
    return 1 if failures else 0
