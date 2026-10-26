import bisect
import gc
import re
from pathlib import Path

import soundfile as sf

from .media import SAMPLE_RATE, is_silent, load_chunk
from .storage import digest, event, model_signature, read_json, write_json


def require_model(path: Path):
    if not path.is_dir() or not (path / "config.json").exists():
        raise ValueError(f"本地模型缺失: {path}；请先按 docs/MODELS.md 下载。")
    if not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")):
        raise ValueError(f"模型权重不完整: {path}")


class ASREngine:
    """One persistent engine per GPU process. Imports CUDA only after worker device isolation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.speaker_pipeline = None
        self.identity = digest({
            "asr": model_signature(cfg.asr_model),
            "aligner": model_signature(cfg.aligner_model) if cfg.word_timestamps else None,
            "language": cfg.language, "timestamps": cfg.word_timestamps,
            "tokens": cfg.max_new_tokens, "attention": cfg.attention,
            "glossary": cfg.glossary if cfg.asr_use_glossary else "", "implementation": 3,
        })

    def load(self):
        if self.model is not None:
            return
        require_model(self.cfg.asr_model)
        if self.cfg.word_timestamps:
            require_model(self.cfg.aligner_model)
        import torch
        from qwen_asr import Qwen3ASRModel
        if not torch.cuda.is_available():
            raise RuntimeError("没有可用 CUDA GPU。当前项目不在 CPU 上执行真实 ASR。")
        kwargs = {"dtype": torch.bfloat16, "device_map": "cuda:0", "local_files_only": True,
                  "attn_implementation": self.cfg.attention,
                  "max_inference_batch_size": self.cfg.batch_size, "max_new_tokens": self.cfg.max_new_tokens}
        if self.cfg.word_timestamps:
            kwargs.update(forced_aligner=str(self.cfg.aligner_model), forced_aligner_kwargs={
                "dtype": torch.bfloat16, "device_map": "cuda:0", "local_files_only": True,
                "attn_implementation": self.cfg.attention,
            })
        self.model = Qwen3ASRModel.from_pretrained(str(self.cfg.asr_model), **kwargs)

    def infer(self, arrays, job):
        import torch
        try:
            self.load()
            return self.model.transcribe(
                audio=[(a, SAMPLE_RATE) for a in arrays],
                context=(f"以下是可能涉及的术语，按实际语音转写：\n{self.cfg.glossary}"
                         if self.cfg.asr_use_glossary and self.cfg.glossary else ""),
                language=None if self.cfg.language.lower() == "auto" else self.cfg.language,
                return_time_stamps=self.cfg.word_timestamps,
            )
        except torch.cuda.OutOfMemoryError:
            pass
        # Leave the exception scope first: its traceback may otherwise keep failed-batch CUDA tensors alive.
        gc.collect()
        torch.cuda.empty_cache()
        if len(arrays) == 1:
            raise RuntimeError("单片音频仍显存不足：请减小 ASR_CHUNK_SECONDS，或关闭时间戳/说话人功能。")
        event(job, "asr", "显存不足，自动缩小当前批次", batch=len(arrays))
        middle = len(arrays) // 2
        return self.infer(arrays[:middle], job) + self.infer(arrays[middle:], job)

    def transcribe(self, job: Path, plan: dict) -> dict:
        key = digest({"plan": plan["key"], "engine": self.identity})
        target = job / "asr.json"
        if target.exists() and (cached := read_json(target)).get("key") == key:
            event(job, "asr", "复用已完成的语音识别")
            return cached
        chunks = plan["chunks"]
        cache_dir = job / "cache" / "asr" / key[:20]
        cache_dir.mkdir(parents=True, exist_ok=True)
        results, pending, arrays = {}, [], []

        def flush():
            if not pending:
                return
            output = self.infer(arrays, job)
            if len(output) != len(pending):
                raise ValueError("ASR 返回批次数量不匹配")
            for chunk, response in zip(pending, output):
                words = []
                warnings = []
                if response.time_stamps is not None:
                    for word in response.time_stamps:
                        begin = max(chunk["start"], min(chunk["end"], chunk["start"] + float(word.start_time)))
                        end = max(begin, min(chunk["end"], chunk["start"] + float(word.end_time)))
                        words.append({"text": word.text, "start": begin, "end": end})
                if response.text.strip() and self.cfg.word_timestamps and not words:
                    warnings.append("对齐模型未返回词时间戳，使用音频片段边界")
                # A hard token limit may truncate text: surface this for human inspection rather than hiding it.
                if len(response.text) >= self.cfg.max_new_tokens * 0.85:
                    warnings.append("文本较长，可能接近生成上限，请检查片段结尾")
                result = {**chunk, "text": response.text.strip(), "language": response.language,
                          "words": words, "warnings": warnings}
                write_json(cache_dir / f"{chunk['index']:06d}.json", result)
                results[chunk["index"]] = result
            event(job, "asr", "完成音频批次", completed=len(results), total=len(chunks))
            pending.clear()
            arrays.clear()

        with sf.SoundFile(job / "audio.wav") as audio:
            for chunk in chunks:
                path = cache_dir / f"{chunk['index']:06d}.json"
                if path.exists():
                    results[chunk["index"]] = read_json(path)
                    continue
                waveform = load_chunk(audio, chunk)
                if is_silent(waveform):
                    result = {**chunk, "text": "", "language": "", "words": [], "warnings": ["数字静音"]}
                    write_json(path, result)
                    results[chunk["index"]] = result
                else:
                    pending.append(chunk)
                    arrays.append(waveform)
                if len(pending) >= self.cfg.batch_size:
                    flush()
            flush()
        data = {"key": key, "chunks": [results[i] for i in range(len(chunks))],
                "model": self.cfg.asr_model.name, "word_timestamps": self.cfg.word_timestamps}
        write_json(target, data)
        return data

    def diarize(self, job: Path, audio_meta: dict) -> dict:
        if not self.cfg.diarization:
            return {"key": "disabled", "turns": []}
        key = digest({"audio": audio_meta["key"], "model": model_signature(self.cfg.diarization_model),
                      "speakers": self.cfg.num_speakers, "implementation": 1})
        path = job / "speakers.json"
        if path.exists() and (cached := read_json(path)).get("key") == key:
            return cached
        if not (self.cfg.diarization_model / "config.yaml").exists():
            raise ValueError("说话人模型未下载完整，参见 docs/MODELS.md")
        import torch
        from pyannote.audio import Pipeline
        event(job, "speakers", "正在分析整场说话人，编号在同一视频内保持一致")
        if self.speaker_pipeline is None:
            self.speaker_pipeline = Pipeline.from_pretrained(str(self.cfg.diarization_model))
            self.speaker_pipeline.to(torch.device("cuda:0"))
        waveform, rate = sf.read(job / "audio.wav", dtype="float32", always_2d=True)
        kwargs = {"num_speakers": self.cfg.num_speakers} if self.cfg.num_speakers else {}
        with torch.inference_mode():
            response = self.speaker_pipeline({"waveform": torch.from_numpy(waveform.T.copy()),
                                              "sample_rate": rate}, **kwargs)
        turns = [{"start": turn.start, "end": turn.end, "speaker": speaker}
                 for turn, _, speaker in response.exclusive_speaker_diarization.itertracks(yield_label=True)]
        data = {"key": key, "turns": turns}
        write_json(path, data)
        return data


class SpeakerIndex:
    def __init__(self, turns):
        self.turns = sorted(turns, key=lambda t: t["start"])
        self.ends = [t["end"] for t in self.turns]

    def assign(self, start, end):
        best, overlap = "未区分", 0
        # Exclusive turns have monotonic ends, so only local overlapping turns are scanned.
        for i in range(bisect.bisect_right(self.ends, start), len(self.turns)):
            turn = self.turns[i]
            if turn["start"] >= end:
                break
            amount = min(end, turn["end"]) - max(start, turn["start"])
            if amount > overlap:
                overlap, best = amount, turn["speaker"]
        return best


def aligned_pieces(chunk: dict) -> list[dict]:
    """Keep ASR punctuation/spacing while attaching character/word times from forced alignment."""
    text, words = chunk["text"], chunk["words"]
    if not words:
        return [{"text": text, "start": chunk["start"], "end": chunk["end"]}]
    norm, positions = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            for c in ch.casefold():
                norm.append(c)
                positions.append(i)
    normalized, cursor, starts = "".join(norm), 0, []
    for word in words:
        token = "".join(c for c in word["text"].casefold() if c.isalnum())
        if not token:
            continue
        found = normalized.find(token, cursor)
        if found != cursor:
            # Do not silently drop text if aligner normalization differs from the original ASR output.
            return [{"text": text, "start": chunk["start"], "end": chunk["end"],
                     "alignment_warning": "对齐文字与转写不匹配，保留原文并使用片段时间"}]
        starts.append((positions[found], word))
        cursor = found + len(token)
    if not starts or cursor != len(normalized):
        return [{"text": text, "start": chunk["start"], "end": chunk["end"],
                 "alignment_warning": "对齐覆盖不完整，保留原文并使用片段时间"}]
    pieces = []
    for i, (pos, word) in enumerate(starts):
        begin = 0 if i == 0 else pos
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        pieces.append({"text": text[begin:end], "start": word["start"], "end": word["end"]})
    return pieces


def build_segments(asr: dict, speakers: dict) -> list[dict]:
    index = SpeakerIndex(speakers["turns"])
    segments = []
    for chunk in asr["chunks"]:
        if not chunk["text"].strip():
            continue
        current = None
        for piece in aligned_pieces(chunk):
            speaker = index.assign(piece["start"], piece["end"])
            new = current is None or speaker != current["speaker"] or (
                current["end"] - current["start"] >= 12 and re.search(r"[。！？.!?]\s*$", current["text"]))
            if new:
                current = {"id": f"s{len(segments) + 1:06d}", "start": piece["start"], "end": piece["end"],
                           "text": "", "speaker": speaker, "chunk": chunk["index"],
                           "warnings": list(chunk.get("warnings", []))}
                segments.append(current)
            current["text"] += piece["text"]
            current["end"] = max(current["end"], piece["end"])
            if piece.get("alignment_warning"):
                current["warnings"].append(piece["alignment_warning"])
    for segment in segments:
        segment["text"] = segment["text"].strip()
    return segments
