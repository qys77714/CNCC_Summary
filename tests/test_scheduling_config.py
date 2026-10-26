
import pytest

from cncc_minutes.config import Settings
from cncc_minutes.pipeline import discover, run
from cncc_minutes.storage import output_lock, read_json


def test_shards_disjoint_and_complete(tmp_path):
    for i in range(37):
        (tmp_path / f"会议{i}.mp4").touch()
    shards = [set(discover(tmp_path, shard_index=i, shard_count=4)) for i in range(4)]
    assert len(set.union(*shards)) == 37
    assert sum(map(len, shards)) == 37
    with pytest.raises(ValueError):
        discover(tmp_path, shard_index=4, shard_count=4)


def test_output_lock_excludes_second_writer(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="已有任务"):
            with output_lock(tmp_path):
                pass


def test_env_paths_relative_to_env_and_key_not_in_repr(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "a-private-secret")
    monkeypatch.setenv("INPUT_DIR", "./videos")
    env = tmp_path / ".env"
    env.write_text("OUTPUT_DIR=./results\n", "utf-8")
    cfg = Settings.load(env)
    assert cfg.input_dir == tmp_path / "videos" and cfg.output_dir == tmp_path / "results"
    assert "a-private-secret" not in repr(cfg)


def test_llm_only_reports_missing_transcript_and_continues(cfg):
    cfg.input_dir.mkdir()
    sources = [cfg.input_dir / f"{i}.mp4" for i in range(2)]
    for p in sources:
        p.touch()
    assert run(cfg, sources, llm_only=True) == 1
    manifest = read_json(cfg.output_dir / "last_run.json")
    assert len(manifest["failures"]) == 2 and manifest["succeeded"] == 0


def test_parallel_extract_failure_isolated(cfg):
    import numpy as np
    import soundfile as sf
    cfg.input_dir.mkdir()
    good = cfg.input_dir / "good.wav"
    bad = cfg.input_dir / "bad.mp4"
    sf.write(good, np.ones(16000) * 0.1, 16000)
    bad.write_bytes(b"invalid-media")
    assert run(cfg, [good, bad], until="extract") == 1
    manifest = read_json(cfg.output_dir / "last_run.json")
    assert manifest["succeeded"] == 1 and len(manifest["failures"]) == 1
    assert len(list(cfg.output_dir.glob("*/audio.wav"))) == 1


def test_outer_cuda_allocation_preserved(cfg, monkeypatch):
    import os
    import queue

    from cncc_minutes.pipeline import gpu_worker
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")
    tasks, results = queue.Queue(), queue.Queue()
    tasks.put(None)
    gpu_worker("1", cfg, tasks, results, "extract")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
