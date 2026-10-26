import json
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

from .prompts import CORRECT, SUMMARIZE
from .report import mark_stale, render_outputs
from .schemas import Corrections, Summary, check_sources
from .storage import digest, event, read_json, write_json


def pack(items, budget):
    groups, current, size = [], [], 2
    for item in items:
        length = len(json.dumps(item, ensure_ascii=False)) + 2
        if length > budget:
            raise ValueError(f"单条资料超过字符预算 {budget}，请提高预算或缩小 ASR 分片")
        if current and size + length > budget:
            groups.append(current)
            current, size = [], 2
        current.append(item)
        size += length
    if current:
        groups.append(current)
    return groups


def correction_guard(original: str, proposed: str) -> str | None:
    if original.strip() and not proposed.strip():
        return "纠错结果清空了原文，已保留原文"
    number_pattern = r"\d+(?:\.\d+)?(?:%|％)?"
    if re.findall(number_pattern, original) != re.findall(number_pattern, proposed):
        return "纠错改变了数字，需回听确认，已保留原文"
    negative_pattern = r"不能|不支持|不建议|不是|没有|不需要|未能|无需|不等于|不可"
    if re.findall(negative_pattern, original) != re.findall(negative_pattern, proposed):
        return "纠错改变了关键否定表达，需回听确认，已保留原文"
    if len(proposed) > max(60, len(original) * 1.65) or len(proposed) < len(original) * 0.65:
        return "纠错文本长度变化过大，已保留原文"
    ratio = SequenceMatcher(None, original, proposed, autojunk=False).ratio()
    if len(original) >= 12 and ratio < 0.6:
        return "纠错改写幅度过大，已保留原文"
    return None


def correct_segments(job, raw, cfg, client):
    targets = [{"id": s["id"], "text": s["text"], "speaker": s["speaker"]} for s in raw]
    groups = pack(targets, cfg.correction_chars)
    by_id = {s["id"]: s for s in raw}
    positions = {s["id"]: i for i, s in enumerate(targets)}

    def run_group(i, group):
        start = positions[group[0]["id"]]
        end = start + len(group)

        def validate(response):
            if [x.id for x in response.segments] != [x["id"] for x in group]:
                raise ValueError("纠错输出 ID 必须与 target 完全一致且顺序相同")

        event(job, "correction", "正在校正转写", batch=i + 1, total=len(groups))
        return client.structured(job, "correction", CORRECT, {
            "context": cfg.context, "glossary": cfg.glossary, "target": group,
            "before": targets[max(0, start - 2):start], "after": targets[end:end + 2],
        }, Corrections, validate)

    # 批次彼此独立（上下文取自原始转写），可并行请求；全局并发由 client 的信号量限制。
    with ThreadPoolExecutor(max_workers=min(len(groups), cfg.llm_concurrency)) as pool:
        responses = list(pool.map(lambda pair: run_group(*pair), enumerate(groups)))

    corrected, audit = [], []
    for group, response in zip(groups, responses):
        for item in response.segments:
            original = by_id[item.id]
            rejected = correction_guard(original["text"], item.text)
            corrected.append({**original, "text": original["text"] if rejected else item.text,
                              "original_text": original["text"], "uncertain": item.uncertain or bool(rejected),
                              "correction_reason": rejected or item.reason})
            if item.text != original["text"] or item.uncertain or rejected:
                audit.append({"id": item.id, "start": original["start"], "before": original["text"],
                              "proposed": item.text, "after": corrected[-1]["text"],
                              "reason": rejected or item.reason, "rejected": bool(rejected),
                              "uncertain": corrected[-1]["uncertain"]})
    return corrected, audit


def cited_ids(summary):
    return {source for field in ("overview", "key_points", "decisions", "actions", "questions", "terms", "uncertainties")
            for item in getattr(summary, field) for source in item.sources}


def summarize_video(job: Path, cfg, client):
    mark_stale(job)
    raw_doc = read_json(job / "transcript.raw.json")
    meta = read_json(job / "audio.json")
    key = digest({"raw": raw_doc, "identity": cfg.llm_identity(), "context": cfg.context, "glossary": cfg.glossary,
                  "budgets": [cfg.correction_chars, cfg.summary_chars, cfg.reduce_chars],
                  "prompts": [CORRECT, SUMMARIZE], "speaker_names": cfg.speaker_names.get(meta["source"], {}),
                  "implementation": 5})
    completed = job / "summary.json"
    # Never leave an old report looking current during an unsuccessful rebuild.
    write_json(job / "report_status.json", {"status": "processing", "key": key})
    if completed.exists() and (cached := read_json(completed)).get("key") == key:
        corrected = read_json(job / "transcript.corrected.json")["segments"]
        audit = read_json(job / "corrections.json")
        render_outputs(job, raw_doc, corrected, audit, cached, meta, cfg)
        event(job, "complete", "复用缓存并重新生成纪要")
        write_json(job / "report_status.json", {"status": "complete", "key": key})
        return
    raw = raw_doc["segments"]
    if not raw:
        raise ValueError("整场未识别到文字。请检查音轨、音量及模型输出。")
    corrected_key = digest({"raw": raw_doc, "identity": cfg.llm_identity(), "prompt": CORRECT,
                            "context": cfg.context, "glossary": cfg.glossary, "budget": cfg.correction_chars,
                            "guard_version": 2})
    corrected_path = job / "transcript.corrected.json"
    if corrected_path.exists() and (cached := read_json(corrected_path)).get("key") == corrected_key:
        corrected, audit = cached["segments"], read_json(job / "corrections.json")
    else:
        corrected, audit = correct_segments(job, raw, cfg, client)
        # Audit first: corrected cache is only considered complete once both artifacts are committed.
        write_json(job / "corrections.json", audit)
        write_json(corrected_path, {"key": corrected_key, "segments": corrected})
    name_map = cfg.speaker_names.get(meta["source"], {})
    inputs = [{"id": s["id"], "start": s["start"], "end": s["end"], "speaker": name_map.get(s["speaker"], s["speaker"]),
               "text": s["text"], "uncertain": s.get("uncertain", False)} for s in corrected]
    limit = min(24000, (cfg.reduce_chars - 1000) // 2 - 100)
    if limit < 1500:
        raise ValueError("REDUCE_INPUT_CHARS 至少需要 4200，建议 32000")

    def create_summary(payload, allowed, stage):
        def validate(response):
            check_sources(response, allowed)
            if len(response.model_dump_json()) > limit:
                raise ValueError(f"纪要 JSON 超过 {limit} 字符，请精简并保留必要信息")
        return client.structured(job, stage, SUMMARIZE, {
            "context": cfg.context, "output_char_limit": limit,
            "length_guidance": f"整个 JSON 对象不超过 {limit} 字符；内容要充实具体，"
                               f"在预算内充分展开方法、数据、结论与局限，不要只写一句话概括",
            **payload,
        }, Summary, validate)

    chapters, summaries = [], []
    groups = pack(inputs, cfg.summary_chars)
    for i, group in enumerate(groups):
        event(job, "chapters", "正在提炼时间章节", chapter=i + 1, total=len(groups))
        summary = create_summary({"scope": "局部章节：提炼本段所有重要信息，勿把时间窗口误认为独立会议", "segments": group},
                                 {s["id"] for s in group}, "chapters")
        chapters.append({"start": group[0]["start"], "end": group[-1]["end"],
                         "segment_ids": [s["id"] for s in group], "summary": summary.model_dump()})
        summaries.append(summary)
    # Bounded hierarchical reduction; detailed chapter notes remain in the final report.
    level = 0
    while len(summaries) > 1:
        level += 1
        groups = pack([s.model_dump() for s in summaries], cfg.reduce_chars)
        if len(groups) >= len(summaries):
            raise ValueError("无法合并纪要，请增大 REDUCE_INPUT_CHARS")
        reduced = []
        for i, group in enumerate(groups):
            allowed = set().union(*(cited_ids(Summary.model_validate(s)) for s in group))
            event(job, "reduce", "正在跨章节合并和去重", level=level, batch=i + 1, total=len(groups))
            reduced.append(create_summary({"scope": "综合纪要：覆盖所有输入主题，保留条件与分歧，去重，不扩大结论；"
                                                    "key_points 覆盖每个重要主题的方法、架构、实验条件、数据、结论与局限，"
                                                    "每条可写多句，整体内容应当充实",
                                           "chapter_notes": group}, allowed, "reduce"))
        summaries = reduced
    result = {"key": key, "title": Path(meta["source"]).stem, "summary": summaries[0].model_dump(),
              "chapters": chapters, "asr_model": raw_doc.get("model", "unknown"), "llm_model": cfg.llm_model}
    write_json(completed, result)
    render_outputs(job, raw_doc, corrected, audit, result, meta, cfg)
    write_json(job / "report_status.json", {"status": "complete", "key": key})
    event(job, "complete", "会议纪要已生成", report=str(job / "minutes.md"))
