import json
from dataclasses import replace

import httpx

from .llm import LLMClient
from .report import render_index, srt_text, transcript_markdown
from .storage import atomic_text, output_lock, write_json
from .summarize import summarize_video


def mock_handler(request):
    """A deterministic simulated service, never used for real meeting inference."""
    body = json.loads(request.content)
    payload = json.loads(body["messages"][1]["content"])
    if "segments" in payload:
        groups = {}
        for segment in payload["segments"]:
            title = "遗忘机制与噪声控制" if "噪声" in segment["text"] else "记忆的写入与检索"
            group = groups.setdefault(title, {"title": title, "content": "", "segment_ids": []})
            group["content"] += segment["text"] + "\n\n"
            group["segment_ids"].append(segment["id"])
        output = {"topics": list(groups.values())}
    elif "catalog" in payload:
        groups = {}
        for entry in payload["catalog"]:
            group = groups.setdefault(entry["title"], {
                "title": entry["title"], "focus": "详细解释此主题讨论的机制、例子与限制。", "material_ids": []})
            group["material_ids"].extend(entry["material_ids"])
        output = {"title": "模拟示例：智能体记忆的设计", "introduction": "", "topics": list(groups.values())}
    else:
        prefix = f"### 进一步讨论（{payload['part']}）" if payload["continuation"] else f"## {payload['theme']}"
        content = "\n\n".join(note["content"].strip() for note in payload["materials"])
        supplement = ""
        if payload["allow_knowledge"]:
            supplement = (
                "\n\n从通用原理看，遗忘不一定是物理删除，也可以通过降低旧记录的检索权重来实现。"
                "这是帮助理解本主题的背景解释。"
                if "噪声" in payload["theme"] else
                "\n\n作为背景，相似度通常衡量表示空间中的接近程度，任务价值则取决于信息能否帮助当前决策；"
                "两者可以相关，但不能直接等同。"
            )
        markdown = f"{prefix}\n\n{content}{supplement}"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": markdown}}], "usage": {"total_tokens": 0}})
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(output, ensure_ascii=False)}}], "usage": {"total_tokens": 0}})


def generate_demo(cfg):
    cfg = replace(cfg, output_dir=cfg.root / "examples" / "demo", llm_base_url="https://simulation.invalid/v1",
                  llm_model="SIMULATED-NO-REAL-INFERENCE", llm_api_key="simulation", diarization=False)
    job = cfg.output_dir / "simulated-meeting"
    with output_lock(cfg.output_dir):
        job.mkdir(parents=True, exist_ok=True)
        write_json(job / "audio.json", {"source": "模拟示例-不对应真实视频.wav", "key": "demo-v2",
                                       "duration": 75, "frames": 1200000, "sample_rate": 16000})
        segments = [
            {"id": "s000001", "start": 0.0, "end": 25.0, "speaker": "未区分", "warnings": [],
             "text": "这是一份模拟资料。智能体的长期记忆需要把历史交互转成可检索的信息，"
                     "而不是把全部对话持续堆进当前上下文。写入时应提取与后续任务有关的事实，"
                     "检索时再结合当前问题筛选相关记录。"},
            {"id": "s000002", "start": 25.0, "end": 50.0, "speaker": "未区分", "warnings": [],
             "text": "模拟讨论接着提出，噪声记忆可能让系统持续召回过时或无关的信息。"
                     "遗忘机制需要考虑时效性、任务相关性与更新频率；"
                     "评估也应检查错误记忆对后续推理的影响，不能只计算记忆条数。"},
            {"id": "s000003", "start": 50.0, "end": 75.0, "speaker": "未区分", "warnings": [],
             "text": "模拟讨论最后回到检索：相似度高不一定意味着有助于当前决策。"
                     "应结合任务目标筛选，再将命中的历史记录交给模型推理。"},
        ]
        write_json(job / "transcript.raw.json", {"source": "模拟示例-不对应真实视频.wav",
                                               "model": "SIMULATED-NO-ASR", "key": "demo-v2", "segments": segments})
        atomic_text(job / "transcript.raw.md", "# 模拟原始转写\n\n" + transcript_markdown(segments, {}))
        atomic_text(job / "subtitles.raw.srt", srt_text(segments))
        client = LLMClient(cfg, transport=httpx.MockTransport(mock_handler))
        try:
            summarize_video(job, cfg, client)
        finally:
            client.close()
        render_index(cfg.output_dir)
    return job / "minutes.md"
