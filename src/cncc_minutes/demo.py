import json
from dataclasses import replace

import httpx

from .llm import LLMClient
from .report import render_index
from .storage import output_lock, write_json
from .summarize import summarize_video


def mock_handler(request):
    """Deterministic fixture server, shared with the CPU integration tests."""
    body = json.loads(request.content)
    payload = json.loads(body["messages"][1]["content"])
    if "target" in payload:
        output = {"segments": [{"id": s["id"], "text": s["text"].replace("检索增墙", "检索增强"),
                                "uncertain": False, "reason": "修正术语同音字" if "检索增墙" in s["text"] else ""}
                               for s in payload["target"]]}
    else:
        if "segments" in payload:
            ids = [s["id"] for s in payload["segments"]]
        else:
            ids = [s for note in payload["chapter_notes"] for item in note["overview"] for s in item["sources"]]
        points = [{"text": "演示资料：长期记忆可采用检索增强，并设计遗忘机制。", "sources": ids[:1]}]
        if len(ids) > 1:
            points.append({"text": "演示资料：评估应考虑噪声记忆的影响。", "sources": ids[-1:]})
        output = {"title": "模拟：智能体记忆讨论", "overview": points, "key_points": points,
                  "decisions": [], "actions": [], "questions": [], "terms": [], "uncertainties": []}
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(output, ensure_ascii=False)}}], "usage": {"total_tokens": 0}})


def generate_demo(cfg):
    cfg = replace(cfg, output_dir=cfg.root / "examples" / "demo", llm_base_url="https://simulation.invalid/v1",
                  llm_model="SIMULATED-NO-REAL-INFERENCE", llm_api_key="simulation", diarization=False)
    job = cfg.output_dir / "simulated-meeting"
    with output_lock(cfg.output_dir):
        job.mkdir(parents=True, exist_ok=True)
        write_json(job / "audio.json", {"source": "模拟示例-不对应真实视频.wav", "key": "demo-v1",
                                       "duration": 50, "frames": 800000, "sample_rate": 16000})
        segments = [
            {"id": "s000001", "start": 0.0, "end": 25.0, "speaker": "未区分", "warnings": [],
             "text": "这是模拟示例。长期记忆可以通过检索增墙实现，同时需要设计遗忘机制。"},
            {"id": "s000002", "start": 25.0, "end": 50.0, "speaker": "未区分", "warnings": [],
             "text": "评估时需要考虑噪声记忆的影响。本段没有明确执行任务。"},
        ]
        write_json(job / "transcript.raw.json", {"source": "模拟示例-不对应真实视频.wav",
                                               "model": "SIMULATED-NO-ASR", "key": "demo-v1", "segments": segments})
        client = LLMClient(cfg, transport=httpx.MockTransport(mock_handler))
        try:
            summarize_video(job, cfg, client)
        finally:
            client.close()
        render_index(cfg.output_dir)
    return job / "minutes.md"
