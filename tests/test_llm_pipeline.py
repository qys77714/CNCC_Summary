import json
from dataclasses import replace

import httpx
import pytest

from cncc_minutes.demo import mock_handler
from cncc_minutes.llm import LLMClient
from cncc_minutes.schemas import Corrections, Summary, check_sources
from cncc_minutes.storage import read_json
from cncc_minutes.summarize import correct_segments, correction_guard, pack, summarize_video


def test_correction_refuses_deletion_and_invention():
    assert correction_guard("原文明确指出不能这样部署，需要验证。", "")
    assert correction_guard("原文明确指出不能这样部署，需要验证。", "已经通过生产验证，可以全面部署，无需额外测试。")
    assert correction_guard("我们使用检索增墙，并配合遗忘策略。", "我们使用检索增强，并配合遗忘策略。") is None
    assert correction_guard("提升 3% 的效果。", "提升 30% 的效果。")
    assert correction_guard("这个模型不能部署在当前服务器上。", "这个模型可以部署在当前服务器上。")


def test_pack_order_and_budget():
    items = [{"id": str(i), "text": "长文本" * 20} for i in range(100)]
    groups = pack(items, 500)
    assert [item for group in groups for item in group] == items
    assert all(len(json.dumps(g, ensure_ascii=False)) <= 500 for g in groups)


def test_end_to_end_report_citations_cache_and_invalidation(cfg, raw_job):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        report = (raw_job / "minutes.md").read_text("utf-8")
        assert "检索增强" in report and "检索增墙" not in report
        assert "## 内容概览" in report and "## 核心观点与技术要点" in report
        # 精简版纪要不包含资料、导航、章节、待办、质量与附录
        for removed in ("会议资料", "快速导航", "时间章节", "待办", "质量与待核对", "附录"):
            assert removed not in report
        assert len(read_json(raw_job / "corrections.json")) == 3
        assert "test-secret" not in "".join(p.read_text("utf-8") for p in raw_job.rglob("*.json"))
        initial_calls = len(calls)
        summarize_video(raw_job, cfg, client)
        assert len(calls) == initial_calls
        cfg.context = "测试：新的会议背景"
        summarize_video(raw_job, cfg, client)
        assert len(calls) > initial_calls
    finally:
        client.close()


def test_summary_rejects_nonexistent_sources():
    summary = Summary(title="测试", overview=[{"text": "事实", "sources": ["wrong"]}],
                      key_points=[], decisions=[], actions=[], questions=[], terms=[], uncertainties=[])
    with pytest.raises(ValueError, match="不存在"):
        check_sources(summary, {"s000001"})


def test_llm_retries_429_then_caches(cfg, raw_job):
    calls, delays = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=delays.append)
    payload = {"target": [{"id": "s000001", "text": "测试"}]}
    try:
        first = client.structured(raw_job, "test", "只返回 JSON", payload, Corrections)
        second = client.structured(raw_job, "test", "只返回 JSON", payload, Corrections)
        assert first == second and len(calls) == 2 and delays[0] >= 2
    finally:
        client.close()


def test_invalid_or_truncated_json_never_produces_success(cfg, raw_job):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]})
    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        with pytest.raises(RuntimeError, match="仍失败"):
            summarize_video(raw_job, cfg, client)
        assert not (raw_job / "minutes.md").exists()
    finally:
        client.close()


def test_correction_cannot_drop_ids(cfg, raw_job):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"segments":[]}'}}]})
    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        with pytest.raises(RuntimeError, match="ID"):
            correct_segments(raw_job, read_json(raw_job / "transcript.raw.json")["segments"], cfg, client)
    finally:
        client.close()


def test_hierarchical_reduction_is_bounded(cfg, raw_job):
    from cncc_minutes.storage import write_json
    cfg = replace(cfg, summary_chars=1000, correction_chars=1000, reduce_chars=6000)
    raw = read_json(raw_job / "transcript.raw.json")
    raw["segments"] *= 20
    for i, segment in enumerate(raw["segments"]):
        raw["segments"][i] = {**segment, "id": f"s{i+1:06d}", "start": i*25, "end": (i+1)*25}
    write_json(raw_job / "transcript.raw.json", raw)
    counts = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        counts.append("reduce" if "chapter_notes" in payload else "other")
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        result = read_json(raw_job / "summary.json")
        assert len(result["chapters"]) > 1 and "reduce" in counts
        assert len(read_json(raw_job / "transcript.corrected.json")["segments"]) == 60
    finally:
        client.close()


def test_speaker_name_update_regenerates_summary_not_correction(cfg, raw_job):
    calls = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append("correction" if "target" in payload else "summary")
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        before = calls.count("correction"), calls.count("summary")
        cfg.speaker_names = {"中文 会议.mp4": {"未区分": "人工确认姓名"}}
        summarize_video(raw_job, cfg, client)
        assert calls.count("correction") == before[0] and calls.count("summary") > before[1]
        # 精简版 minutes.md 不含逐字稿；说话人姓名体现在校正逐字稿中
        assert "人工确认姓名" in (raw_job / "transcript.md").read_text("utf-8")
    finally:
        client.close()


def test_failed_rebuild_does_not_expose_old_report_as_current(cfg, raw_job):
    client = LLMClient(cfg, transport=httpx.MockTransport(mock_handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
    finally:
        client.close()
    cfg.context = "force-new-request"
    client = LLMClient(cfg, transport=httpx.MockTransport(lambda request: httpx.Response(401)),
                       sleep=lambda _: None)
    try:
        with pytest.raises(RuntimeError, match="401"):
            summarize_video(raw_job, cfg, client)
        assert not (raw_job / "minutes.md").exists()
        assert (raw_job / "minutes.previous.md").exists()
    finally:
        client.close()
