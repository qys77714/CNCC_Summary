import json
from dataclasses import replace

import httpx
import pytest

from cncc_minutes.demo import mock_handler
from cncc_minutes.llm import LLMClient
from cncc_minutes.schemas import ReadingNotes, TopicPlan
from cncc_minutes.storage import read_json, write_json
from cncc_minutes.summarize import filter_segments, pack, summarize_video, validate_markdown, validate_plan


def test_pack_order_and_budget():
    items = [{"id": str(i), "text": "长文本" * 20} for i in range(100)]
    groups = pack(items, 500)
    assert [item for group in groups for item in group] == items
    assert all(len(json.dumps(g, ensure_ascii=False)) <= 500 for g in groups)


def test_thematic_report_no_correction_no_template_and_cache(cfg, raw_job):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return mock_handler(request)

    original = (raw_job / "transcript.raw.json").read_bytes()
    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        report = (raw_job / "minutes.md").read_text("utf-8")
        assert "## 记忆的写入与检索" in report
        for removed in ("会议资料", "快速导航", "时间章节", "待办", "质量与待核对", "附录", "#s000", "<a "):
            assert removed not in report
        assert not (raw_job / "corrections.json").exists()
        assert not (raw_job / "transcript.corrected.json").exists()
        assert (raw_job / "transcript.raw.json").read_bytes() == original
        assert "test-secret" not in "".join(p.read_text("utf-8") for p in raw_job.rglob("*.json"))
        assert all("target" not in json.loads(c["messages"][1]["content"]) for c in calls)
        initial_calls = len(calls)
        summarize_video(raw_job, cfg, client)
        assert len(calls) == initial_calls
        cfg.context = "新的会议背景"
        summarize_video(raw_job, cfg, client)
        assert len(calls) > initial_calls
    finally:
        client.close()


def test_old_summary_and_corrected_transcript_not_reused(cfg, raw_job):
    write_json(raw_job / "summary.json", {"key": "old", "summary": {"overview": "旧版"}})
    write_json(raw_job / "transcript.corrected.json", {"segments": [{"text": "不应读取的旧纠错稿"}]})
    client = LLMClient(cfg, transport=httpx.MockTransport(mock_handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        result = read_json(raw_job / "summary.json")
        assert result["format"] == "thematic-v2"
        assert "不应读取的旧纠错稿" not in result["markdown"]
    finally:
        client.close()


def test_plan_requires_every_material_once():
    plan = TopicPlan(title="主题", introduction="", topics=[
        {"title": "检索", "focus": "解释检索", "material_ids": ["n000001"]}])
    validate_plan(plan, {"n000001"})
    with pytest.raises(ValueError, match="全部"):
        validate_plan(plan, {"n000001", "n000002"})
    plan.topics[0].material_ids.append("n000001")
    with pytest.raises(ValueError, match="全部"):
        validate_plan(plan, {"n000001"})


def test_llm_retries_429_then_caches(cfg, raw_job):
    calls, delays = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=delays.append)
    payload = {"segments": [{"id": "s000001", "text": "测试"}]}
    try:
        first = client.structured(raw_job, "test", "只返回 JSON", payload, ReadingNotes)
        second = client.structured(raw_job, "test", "只返回 JSON", payload, ReadingNotes)
        assert first == second and len(calls) == 2 and delays[0] >= 2
    finally:
        client.close()


def test_truncated_response_never_produces_success(cfg, raw_job):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]})
    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        with pytest.raises(RuntimeError, match="仍失败"):
            summarize_video(raw_job, cfg, client)
        assert not (raw_job / "minutes.md").exists()
        assert read_json(raw_job / "report_status.json")["status"] == "failed"
    finally:
        client.close()


def test_notes_cannot_reference_nonexistent_segments(cfg, raw_job):
    def handler(request):
        output = {"topics": [{"title": "主题", "content": "详细材料", "segment_ids": ["wrong"]}]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(output)}}]})
    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        with pytest.raises(RuntimeError, match="不存在"):
            summarize_video(raw_job, cfg, client)
    finally:
        client.close()


def test_cross_window_themes_keep_all_continuations_without_final_compression(cfg, raw_job):
    cfg = replace(cfg, summary_chars=1000, reduce_chars=6000)
    raw = read_json(raw_job / "transcript.raw.json")
    base = raw["segments"][0]
    raw["segments"] = [{**base, "id": f"s{i+1:06d}", "text": base["text"] + f"实验条件{i}。",
                        "start": i * 25, "end": (i + 1) * 25} for i in range(240)]
    write_json(raw_job / "transcript.raw.json", raw)
    stages = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        stages.append(payload)
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        result = read_json(raw_job / "summary.json")
        assert len(result["materials"]) > 1 and len(result["topics"]) == 1
        writing = [p for p in stages if "theme" in p]
        assert len(writing) > 1 and writing[-1]["continuation"]
        assert "实验条件0。" in result["markdown"] and "实验条件239。" in result["markdown"]
        assert len([p for p in stages if "catalog" in p]) > 1
    finally:
        client.close()


def test_speaker_names_and_style_are_forwarded(cfg, raw_job):
    cfg.speaker_names = {"中文 会议.mp4": {"未区分": "人工确认姓名"}}
    cfg.allow_knowledge = False
    cfg.summary_style = "展开技术原理，必要时使用比较表格"
    calls = []

    def handler(request):
        calls.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
        return mock_handler(request)

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        summarize_video(raw_job, cfg, client)
        assert any(p.get("segments", [{}])[0].get("speaker") == "人工确认姓名" for p in calls if "segments" in p)
        writing = next(p for p in calls if "theme" in p)
        assert writing["allow_knowledge"] is False and writing["style"] == cfg.summary_style
        assert "从通用原理看" not in (raw_job / "minutes.md").read_text("utf-8")
    finally:
        client.close()


def test_markdown_keeps_tables_math_and_code_and_ignores_json_mode(cfg, raw_job):
    text = "## 机制对比\n\n| 路线 | 代价 |\n| --- | --- |\n| A | 小 |\n\n$x < y$\n\n```python\nx = 1\n```"
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})

    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        assert client.markdown(raw_job, "writing", "写 Markdown", {}, validate_markdown) == text
        assert "response_format" not in calls[0]
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


def test_glossary_echo_filtered_but_substantive_discussion_retained():
    glossary = "CNCC：中国计算机大会\nCCF：中国计算机学会\nASR：自动语音识别\nLLM：大语言模型\nRAG：检索增强生成\nAgent：智能体"
    echo = glossary.replace("\n", "，")
    real = "这里讨论 RAG 与 Agent 的关系：检索增强生成能够补充证据，但必须结合智能体当前的任务目标进行筛选。"
    segments = [{"id": "s1", "text": echo}, {"id": "s2", "text": real}, {"id": "s3", "text": echo}]
    kept, excluded = filter_segments(segments, glossary)
    assert [s["id"] for s in kept] == ["s2"]
    assert [s["reason"] for s in excluded] == ["glossary_echo", "glossary_echo"]


def test_no_substantive_notes_fails_instead_of_filling_with_knowledge(cfg, raw_job):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"topics":[]}'}}]})
    client = LLMClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    try:
        with pytest.raises(ValueError, match="实质主题"):
            summarize_video(raw_job, cfg, client)
        assert not (raw_job / "minutes.md").exists()
    finally:
        client.close()


@pytest.mark.parametrize("text", ["## 待办事项\n\n任务", "## 附录\n\n逐字稿", "原文见 s000001。"])
def test_report_rejects_legacy_scaffolding(text):
    with pytest.raises(ValueError):
        validate_markdown(text)
