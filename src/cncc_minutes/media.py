import bisect
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .storage import digest, event, read_json, write_json

SAMPLE_RATE = 16000
MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".flac"}


def ffmpeg_binary(configured="") -> str:
    if configured:
        if not Path(configured).is_file() and not shutil.which(configured):
            raise ValueError(f"FFMPEG_PATH 不存在: {configured}")
        return configured
    if executable := shutil.which("ffmpeg"):
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def prepare_audio(source: Path, job: Path, signature: dict, cfg) -> dict:
    audio_path = job / "audio.wav"
    meta_path = job / "audio.json"
    key = digest({"source": signature, "format": "pcm_s16le-mono-16000-v1"})
    if meta_path.exists() and audio_path.exists():
        meta = read_json(meta_path)
        if meta.get("key") == key:
            info = sf.info(audio_path)
            if info.frames == meta["frames"] and info.samplerate == SAMPLE_RATE and info.channels == 1:
                return meta
    event(job, "audio", "正在导出 16kHz 单声道音频")
    temp = job / "audio.partial.wav"
    cmd = [ffmpeg_binary(cfg.ffmpeg_path), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
           "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-c:a", "pcm_s16le", "-rf64", "auto", str(temp)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"音频导出失败（确认视频有音轨）: {result.stderr[-3000:]}")
    info = sf.info(temp)
    if info.frames == 0:
        raise ValueError("音轨为空")
    os.replace(temp, audio_path)
    meta = {"key": key, "source": source.name, "frames": info.frames, "sample_rate": SAMPLE_RATE,
            "duration": info.duration}
    write_json(meta_path, meta)
    return meta


def silence_centers(audio: Path, binary: str) -> list[float]:
    result = subprocess.run([binary, "-hide_banner", "-nostdin", "-i", str(audio), "-af",
                             "silencedetect=noise=-38dB:d=0.25", "-f", "null", "-"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"静音分析失败: {result.stderr[-1500:]}")
    centers, start = [], None
    for line in result.stderr.splitlines():
        if m := re.search(r"silence_start: ([\d.]+)", line):
            start = float(m[1])
        if m := re.search(r"silence_end: ([\d.]+)", line):
            if start is not None:
                centers.append((start + float(m[1])) / 2)
                start = None
    return centers


def plan_chunks(frames: int, seconds: float, centers: list[float]) -> list[dict]:
    """Contiguous half-open sample intervals: no overlap, dropped pauses or rounded boundary drift."""
    pauses = sorted(round(t * SAMPLE_RATE) for t in centers)
    maximum = round(seconds * SAMPLE_RATE)
    window = min(round(6 * SAMPLE_RATE), maximum // 3)
    chunks, start = [], 0
    while start < frames:
        target = min(start + maximum, frames)
        end = target
        if target < frames:
            i = bisect.bisect_right(pauses, target) - 1
            if i >= 0 and target - window <= pauses[i] <= target:
                end = pauses[i]
        chunks.append({"index": len(chunks), "start_frame": start, "end_frame": end,
                       "start": start / SAMPLE_RATE, "end": end / SAMPLE_RATE})
        start = end
    return chunks


def prepare_chunks(job: Path, audio_meta: dict, cfg) -> dict:
    key = digest({"audio": audio_meta["key"], "seconds": cfg.chunk_seconds, "splitter": 1})
    target = job / "chunks.json"
    if target.exists() and (cached := read_json(target)).get("key") == key:
        return cached
    event(job, "split", "在停顿附近切分音频，保留全部时间范围")
    centers = silence_centers(job / "audio.wav", ffmpeg_binary(cfg.ffmpeg_path))
    data = {"key": key, "chunks": plan_chunks(audio_meta["frames"], cfg.chunk_seconds, centers)}
    write_json(target, data)
    return data


def load_chunk(audio: sf.SoundFile, chunk: dict) -> np.ndarray:
    audio.seek(chunk["start_frame"])
    waveform = audio.read(chunk["end_frame"] - chunk["start_frame"], dtype="float32")
    if len(waveform) != chunk["end_frame"] - chunk["start_frame"]:
        raise ValueError("音频分片读取不完整")
    return waveform


def is_silent(waveform: np.ndarray) -> bool:
    # Conservative digital-silence floor only; low-volume speech is still sent to ASR.
    rms = math.sqrt(float(np.mean(waveform.astype("float64") ** 2)))
    return rms < 1e-5 and float(np.max(np.abs(waveform), initial=0)) < 1e-4
