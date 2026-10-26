import html
import os
from pathlib import Path
from urllib.parse import quote

from .storage import atomic_text, digest, read_json, timestamp, write_json


def clean(text):
    return html.escape(str(text).replace("\n", " ").replace("\r", " ")).replace("|", "\\|")


def transcript_markdown(segments, names, media_url=""):
    lines = []
    for s in segments:
        label = f"{timestamp(s['start'])}–{timestamp(s['end'])}"
        clock = f"[{label}]({media_url}#t={s['start']:.2f})" if media_url else label
        name = clean(names.get(s["speaker"], s["speaker"]))
        uncertain = " · 待核对" if s.get("uncertain") or s.get("warnings") else ""
        lines += [f'<a id="{s["id"]}"></a>', f"**{clock} · {name}{uncertain}**", "", clean(s["text"]), ""]
    return "\n".join(lines)


def srt_text(segments):
    return "\n".join(f"{i}\n{timestamp(s['start'], True)} --> {timestamp(s['end'], True)}\n{s['text']}\n"
                     for i, s in enumerate(segments, 1))


def summary_sections(summary, cite, heading=2):
    prefix = "#" * heading
    lines = []
    for field, title in [("overview", "内容概览"), ("key_points", "核心观点与技术要点"),
                         ("decisions", "明确决议"), ("actions", "待办事项"), ("questions", "问答整理"),
                         ("terms", "术语与概念"), ("uncertainties", "分歧、局限与待确认")]:
        lines += [f"{prefix} {title}", ""]
        items = summary[field]
        if not items:
            lines += [{"decisions": "原文未明确形成决议。", "actions": "原文未明确提出可执行待办。",
                       "questions": "本段未识别到明确问答。", "terms": "未提取额外术语解释。",
                       "uncertainties": "未额外提取；请同时查看文末质量提示。"}.get(field, "未提取。"), ""]
            continue
        if field == "actions":
            lines += ["| 事项 | 责任人 | 截止时间 | 原文依据 |", "| --- | --- | --- | --- |"]
            for item in items:
                lines.append(f"| {clean(item['task'])} | {clean(item['owner'] or '未明确')} | "
                             f"{clean(item['deadline'] or '未明确')} | {cite(item['sources'])} |")
        else:
            for item in items:
                if field == "questions":
                    body = f"**问：{clean(item['question'])}** 答：{clean(item['answer'])}"
                elif field == "terms":
                    body = f"**{clean(item['name'])}**：{clean(item['explanation'])}"
                else:
                    body = clean(item["text"])
                lines.append(f"- {body} {cite(item['sources'])}")
        lines.append("")
    return lines


def minutes_sections(summary, heading=2):
    """minutes.md 只保留总结正文：内容概览、核心观点与技术要点、问答整理，不带引用锚点。"""
    prefix = "#" * heading
    lines = []
    for field, title in [("overview", "内容概览"), ("key_points", "核心观点与技术要点"),
                         ("questions", "问答整理")]:
        items = summary[field]
        if not items:
            continue
        lines += [f"{prefix} {title}", ""]
        for item in items:
            if field == "questions":
                body = f"**问：{clean(item['question'])}** 答：{clean(item['answer'])}"
            else:
                body = clean(item["text"])
            lines.append(f"- {body}")
        lines.append("")
    return lines


def render_outputs(job: Path, raw_doc, corrected, audit, result, meta, cfg):
    source = Path(raw_doc["source"])
    if not source.exists():
        source = cfg.input_dir / meta["source"]
    media_url = quote(os.path.relpath(source, job), safe="/:") if source.exists() else ""
    names = cfg.speaker_names.get(meta["source"], {})
    atomic_text(job / "transcript.raw.md", "# 原始 ASR 转写\n\n" + transcript_markdown(raw_doc["segments"], names, media_url))
    transcript = transcript_markdown(corrected, names, media_url)
    atomic_text(job / "transcript.md", "# 校正逐字稿\n\n> 时间戳来自原音频对齐；纠错后的词语没有再次强制对齐。\n\n" + transcript)
    atomic_text(job / "subtitles.raw.srt", srt_text(raw_doc["segments"]))
    atomic_text(job / "subtitles.srt", srt_text(corrected))

    quality = {
        "segments": len(corrected), "duration_seconds": meta["duration"], "corrections": len(audit),
        "rejected_corrections": sum(bool(x["rejected"]) for x in audit),
        "uncertain_segments": [s["id"] for s in corrected if s.get("uncertain")],
        "asr_warnings": [{"id": s["id"], "warnings": s["warnings"]} for s in corrected if s.get("warnings")],
        "diarization_enabled": cfg.diarization,
        "citation_check": "所有引用 ID 均已校验；语义是否支持结论仍需人工抽查",
    }
    write_json(job / "quality.json", quality)
    lines = [f"# {clean(result['title'])}", ""]
    if result["asr_model"].startswith("SIMULATED"):
        lines += ["> 此文件使用模拟文字和模拟响应生成，没有执行 ASR 或真实 LLM 调用，"
                  "也没有对应的音视频文件。", ""]
    lines += minutes_sections(result["summary"])
    atomic_text(job / "minutes.md", "\n".join(lines) + "\n")


def render_index(root: Path):
    lines = ["# 会议纪要索引", "", "各任务的当前状态如下。仅状态为 complete 的纪要属于本次成功结果。", "",
             "| 会议 | 状态 | 纪要 | 音频 | 进度详情 |", "| --- | --- | --- | --- | --- |"]
    for directory in sorted(root.iterdir()):
        status_path = directory / "status.json"
        if not directory.is_dir() or not status_path.exists():
            continue
        status = read_json(status_path)
        meta_path = directory / "audio.json"
        title = read_json(meta_path)["source"] if meta_path.exists() else directory.name
        link = f"[阅读]({quote(directory.name)}/minutes.md)" if status["stage"] == "complete" else "—"
        audio = f"[WAV]({quote(directory.name)}/audio.wav)" if (directory / "audio.wav").exists() else "—"
        lines.append(f"| {clean(title)} | {clean(status['stage'])} | {link} | {audio} | {clean(status['message'])} |")
    atomic_text(root / "README.md", "\n".join(lines) + "\n")


def mark_stale(job: Path, message="上次结果已过期，本轮尚未生成纪要"):
    path = job / "minutes.md"
    if path.exists():
        os.replace(path, job / "minutes.previous.md")
    write_json(job / "report_status.json", {"status": "stale", "reason": message, "marker": digest(message)})
