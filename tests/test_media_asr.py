from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from cncc_minutes.asr import ASREngine, SpeakerIndex, aligned_pieces, build_segments
from cncc_minutes.media import (
    SAMPLE_RATE,
    ffmpeg_binary,
    is_silent,
    plan_chunks,
    prepare_audio,
    prepare_chunks,
)
from cncc_minutes.storage import source_signature


@pytest.mark.parametrize("duration", [0.01, 25, 25.001, 300, 14400])
def test_sample_intervals_cover_whole_audio_exactly(duration):
    frames = round(duration * SAMPLE_RATE)
    chunks = plan_chunks(frames, 25, [24.3, 46.2, 70.5, 120.0])
    assert chunks[0]["start_frame"] == 0 and chunks[-1]["end_frame"] == frames
    assert sum(c["end_frame"] - c["start_frame"] for c in chunks) == frames
    assert all(a["end_frame"] == b["start_frame"] for a, b in zip(chunks, chunks[1:]))
    assert all(0 < c["end_frame"] - c["start_frame"] <= 25 * SAMPLE_RATE for c in chunks)


def test_real_audio_extract_and_cache(cfg, tmp_path):
    source = tmp_path / "原始音频.wav"
    waveform = np.sin(np.arange(48000 * 3) * 2*np.pi*440/48000) * 0.1
    sf.write(source, np.column_stack([waveform, waveform]), 48000)
    job = tmp_path / "job"
    job.mkdir()
    sig = source_signature(source)
    meta = prepare_audio(source, job, sig, cfg)
    audio = sf.info(job / "audio.wav")
    assert audio.channels == 1 and audio.samplerate == SAMPLE_RATE
    assert abs(audio.duration - 3) < 0.01
    before = (job / "audio.wav").stat().st_mtime_ns
    assert prepare_audio(source, job, sig, cfg) == meta
    assert (job / "audio.wav").stat().st_mtime_ns == before
    plan = prepare_chunks(job, meta, cfg)
    assert plan["chunks"][-1]["end_frame"] == audio.frames
    assert Path(ffmpeg_binary()).exists()


def test_only_digital_silence_skipped():
    assert is_silent(np.zeros(1600, dtype=np.float32))
    assert not is_silent(np.full(1600, 0.001, dtype=np.float32))


def test_punctuation_preserved_and_speaker_boundaries():
    chunk = {"index": 0, "start": 0, "end": 5, "text": "你好，Qwen！再见。",
             "words": [{"text": "你", "start": 0, "end": 1}, {"text": "好", "start": 1, "end": 2},
                       {"text": "Qwen", "start": 2, "end": 3},
                       {"text": "再", "start": 3, "end": 4}, {"text": "见", "start": 4, "end": 5}]}
    pieces = aligned_pieces(chunk)
    assert "".join(p["text"] for p in pieces) == chunk["text"]
    result = build_segments({"chunks": [chunk]}, {"turns": [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00"}, {"start": 3, "end": 5, "speaker": "SPEAKER_01"}]})
    assert [s["text"] for s in result] == ["你好，Qwen！", "再见。"]
    assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]


def test_alignment_mismatch_never_drops_words():
    chunk = {"start": 0, "end": 2, "text": "这是完整原文。", "words": [{"text": "错误", "start": 0, "end": 1}]}
    result = aligned_pieces(chunk)
    assert result[0]["text"] == chunk["text"] and "alignment_warning" in result[0]


def test_speaker_overlap_index():
    index = SpeakerIndex([{"start": 1, "end": 3, "speaker": "A"}, {"start": 4, "end": 7, "speaker": "B"}])
    assert index.assign(2.9, 5) == "B"
    assert index.assign(0, 0.5) == "未区分"


def test_asr_chunk_cache_resume_without_gpu(cfg, tmp_path):
    cfg = replace(cfg, word_timestamps=False, batch_size=1)
    job = tmp_path / "asr-job"
    job.mkdir()
    sf.write(job / "audio.wav", np.full(16000 * 4, 0.1, dtype=np.float32), 16000)
    chunks = plan_chunks(16000 * 4, 2, [])
    engine = ASREngine(cfg)
    calls = []

    def infer(arrays, job):
        calls.append(len(arrays))
        return [SimpleNamespace(text="测试中文。", language="Chinese", time_stamps=None) for _ in arrays]

    engine.infer = infer
    result = engine.transcribe(job, {"key": "plan-1", "chunks": chunks})
    assert len(calls) == 2
    (job / "asr.json").unlink()
    result2 = engine.transcribe(job, {"key": "plan-1", "chunks": chunks})
    assert result == result2 and len(calls) == 2
    engine.transcribe(job, {"key": "plan-changed", "chunks": chunks})
    assert len(calls) == 4


def test_qwen_adapter_loads_local_models_on_isolated_cuda(cfg, monkeypatch):
    import sys
    for model in (cfg.asr_model, cfg.aligner_model):
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", "utf-8")
        (model / "model.safetensors").write_bytes(b"fixture-not-a-model")
    received = {}

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            received.update(path=path, **kwargs)
            return cls()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=Model))
    engine = ASREngine(cfg)
    engine.load()
    assert received["device_map"] == "cuda:0" and received["local_files_only"] is True
    assert received["forced_aligner_kwargs"]["local_files_only"] is True
    assert received["forced_aligner"] == str(cfg.aligner_model)


def test_oom_batch_halves_and_preserves_result_order(cfg, tmp_path, monkeypatch):
    import sys
    class OutOfMemoryError(Exception):
        pass
    released, sizes = [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=lambda: released.append(True))))

    class Model:
        def transcribe(self, audio, **kwargs):
            sizes.append(len(audio))
            if len(audio) > 1:
                raise OutOfMemoryError()
            return [float(audio[0][0][0])]

    engine = ASREngine(cfg)
    engine.model = Model()
    result = engine.infer([np.array([1.0]), np.array([2.0])], tmp_path / "oom-job")
    assert result == [1.0, 2.0] and sizes == [2, 1, 1] and released == [True]
