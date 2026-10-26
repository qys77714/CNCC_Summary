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


def render_outputs(job: Path, result):
    """Keep the LLM's Markdown intact; no fixed report sections or HTML escaping."""
    atomic_text(job / "minutes.md", result["markdown"].strip() + "\n")


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
