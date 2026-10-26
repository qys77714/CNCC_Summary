
import pytest

from cncc_minutes.config import Settings


@pytest.fixture
def cfg(tmp_path):
    return Settings(root=tmp_path, input_dir=tmp_path / "video", output_dir=tmp_path / "output",
                    asr_model=tmp_path / "models/asr", aligner_model=tmp_path / "models/aligner",
                    diarization_model=tmp_path / "models/speakers", llm_api_key="test-secret",
                    llm_base_url="https://llm.invalid/v1", llm_model="fixture-model", llm_retries=2)


@pytest.fixture
def raw_job(cfg):
    from cncc_minutes.storage import write_json
    job = cfg.output_dir / "fixture"
    job.mkdir(parents=True)
    cfg.input_dir.mkdir()
    source = cfg.input_dir / "中文 会议.mp4"
    source.write_bytes(b"not-real-media")
    segments = [{"id": f"s{i:06d}", "start": (i-1)*25.0, "end": i*25.0, "speaker": "未区分",
                 "warnings": [], "text": "长期记忆可以通过检索增墙实现，同时需要设计遗忘机制。"}
                for i in range(1, 4)]
    write_json(job / "transcript.raw.json", {"source": str(source), "model": "fixture-asr",
                                           "key": "raw-fixture", "segments": segments})
    write_json(job / "audio.json", {"source": source.name, "key": "audio-fixture", "duration": 75,
                                   "frames": 1200000, "sample_rate": 16000})
    return job
