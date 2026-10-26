import json
import re
from collections import Counter
from pathlib import Path

from .prompts import PLAN, READ, WRITE
from .report import mark_stale, render_outputs
from .schemas import ReadingNotes, TopicPlan
from .storage import digest, event, read_json, write_json

FORMAT_VERSION = "thematic-v2"


def serialized(value):
    return json.dumps(value, ensure_ascii=False)


def pack(items, budget):
    groups, current, size = [], [], 2
    for item in items:
        length = len(serialized(item)) + 2
        if length + 2 > budget:
            raise ValueError(f"单条材料超过字符预算 {budget}，请提高 REDUCE_INPUT_CHARS / SUMMARY_CHARS")
        if current and size + length > budget:
            groups.append(current)
            current, size = [], 2
        current.append(item)
        size += length
    if current:
        groups.append(current)
    return groups


def normalized(text):
    return "".join(c.casefold() for c in text if c.isalnum())


def filter_segments(segments, glossary):
    """Exclude clear glossary echoes and adjacent exact repeats only; leave raw ASR untouched."""
    atoms = set()
    for line in glossary.splitlines():
        if line.lstrip().startswith("#"):
            continue
        atoms.update(normalized(x) for x in re.split(r"[:：]", line) if len(normalized(x)) >= 2)
    atoms = sorted(atoms, key=len, reverse=True)
    kept, excluded, previous = [], [], ""
    for segment in segments:
        text = segment["text"]
        norm = normalized(text)
        remaining, matches = norm, 0
        for atom in atoms:
            if atom in remaining:
                matches += remaining.count(atom)
                remaining = remaining.replace(atom, "")
        definitions = len(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{1,20}\s*[:：]", text))
        glossary_echo = (len(norm) >= 30 and matches >= 6 and len(remaining) <= len(norm) * 0.12
                         and (definitions >= 3 or matches >= 10))
        reason = ""
        if not norm:
            reason = "empty"
        elif glossary_echo:
            reason = "glossary_echo"
        elif len(norm) >= 30 and norm == previous:
            reason = "adjacent_exact_repeat"
        if reason:
            excluded.append({"id": segment["id"], "reason": reason})
        else:
            kept.append(segment)
        previous = norm
    return kept, excluded


def validate_notes(notes, allowed, limit):
    for topic in notes.topics:
        if not set(topic.segment_ids) <= allowed:
            raise ValueError("材料笔记引用了目标输入中不存在的段落 ID")
    if len(serialized(notes.model_dump())) > limit:
        raise ValueError(f"本组材料笔记超过 {limit} 字符，请消除口语重复后保留实质细节")


def validate_plan(plan, allowed):
    assigned = [i for topic in plan.topics for i in topic.material_ids]
    if Counter(assigned) != Counter({i: 1 for i in allowed}):
        raise ValueError("主题规划必须将全部 material_ids 各分配一次，不可丢失、重复或新增")


def validate_markdown(text, continuation=False):
    if not text.strip():
        raise ValueError("主题正文为空")
    if re.search(r"\b[sn]\d{6}\b|\]\([^)]*(?:transcript|#s\d)", text):
        raise ValueError("正文不得包含片段编号或转写引用")
    if re.search(r"(?m)^#{1,6}\s*(?:\d+[.、]\s*)?(?:会议资料|快速导航|会议导航|待办事项|"
                 r"行动项|附录|质量与待核对|质量报告|完整校正逐字稿|时间章节)(?:\s|[：:]|$)", text):
        raise ValueError("正文包含应删除的固定栏目，请只写主题内容")
    if continuation and re.search(r"(?m)^#{1,2}\s+", text):
        raise ValueError("续写不得重复全文或主题标题，请使用下级小标题或直接写正文")


def plan_topics(job, cfg, client, materials):
    catalog = [{"title": note["title"], "focus": note["content"][:450], "material_ids": [note["id"]]}
               for note in materials]
    previous_size = None
    for level in range(12):
        groups = pack(catalog, cfg.reduce_chars)
        plans = []
        for i, group in enumerate(groups):
            allowed = {mid for item in group for mid in item["material_ids"]}
            event(job, "outline", "正在归并全场主题", level=level + 1, batch=i + 1, total=len(groups))
            plans.append(client.structured(
                job, "outline", PLAN, {"context": cfg.context, "catalog": group},
                TopicPlan, lambda plan, ids=allowed: validate_plan(plan, ids),
            ))
        if len(plans) == 1:
            return plans[0]
        catalog = [topic.model_dump() for plan in plans for topic in plan.topics]
        size = len(serialized(catalog))
        if previous_size is not None and size >= previous_size:
            raise ValueError("主题目录仍过大且未能合并，请提高 REDUCE_INPUT_CHARS")
        previous_size = size
    raise ValueError("主题目录层级过多，请提高 REDUCE_INPUT_CHARS")


def write_topic(job, cfg, client, topic, materials, raw_by_id, all_topics):
    selected = [materials[mid] for mid in topic.material_ids]
    # Split oversized themes into continuations, then keep all detailed prose. No final lossy reduction.
    groups = pack(selected, cfg.reduce_chars - 2000)
    parts = []
    for i, group in enumerate(groups):
        excerpt_budget = max(0, cfg.reduce_chars - 2000 - len(serialized(group)))
        excerpts = []
        for sid in dict.fromkeys(s for material in group for s in material["segment_ids"]):
            segment = raw_by_id[sid]
            excerpt = {"speaker": segment["speaker"], "text": segment["text"]}
            size = len(serialized(excerpt)) + 2
            if size <= excerpt_budget:
                excerpts.append(excerpt)
                excerpt_budget -= size
        previous = "\n\n".join(parts)
        event(job, "writing", "正在详细撰写主题", topic=topic.title, part=i + 1, total=len(groups))
        markdown = client.markdown(job, "writing", WRITE, {
            "context": cfg.context, "theme": topic.title, "focus": topic.focus,
            "all_topics": all_topics, "materials": group, "source_excerpts": excerpts,
            "allow_knowledge": cfg.allow_knowledge, "style": cfg.summary_style,
            "continuation": i > 0, "part": i + 1, "total_parts": len(groups),
            "written_tail": previous[-1600:],
            "written_headings": re.findall(r"(?m)^#{2,6}\s+(.+)$", previous),
        }, lambda text, continued=i > 0: validate_markdown(text, continued))
        if i == 0 and not re.search(r"(?m)^##\s+", markdown):
            markdown = f"## {topic.title}\n\n{markdown}"
        parts.append(markdown)
    return "\n\n".join(parts)


def summarize_video(job: Path, cfg, client):
    mark_stale(job)
    raw_doc = read_json(job / "transcript.raw.json")
    meta = read_json(job / "audio.json")
    names = cfg.speaker_names.get(meta["source"], {})
    key = digest({
        "format": FORMAT_VERSION, "raw": raw_doc, "identity": cfg.llm_identity(),
        "context": cfg.context, "style": cfg.summary_style, "allow_knowledge": cfg.allow_knowledge,
        "budgets": [cfg.summary_chars, cfg.reduce_chars], "glossary_filter": cfg.glossary,
        "speaker_names": names, "prompts": [READ, PLAN, WRITE],
    })
    completed = job / "summary.json"
    write_json(job / "report_status.json", {"status": "processing", "key": key})
    try:
        if completed.exists() and (cached := read_json(completed)).get("key") == key:
            render_outputs(job, cached)
        else:
            segments, excluded = filter_segments(raw_doc["segments"], cfg.glossary)
            if not segments:
                raise ValueError("过滤空白和明显词表复读后没有有效转写，请检查音轨或重新运行 ASR")
            inputs = [{"id": s["id"], "speaker": names.get(s["speaker"], s["speaker"]), "text": s["text"]}
                      for s in segments]
            by_id = {s["id"]: s for s in inputs}
            positions = {s["id"]: i for i, s in enumerate(inputs)}
            groups = pack(inputs, cfg.summary_chars)
            materials = []
            note_limit = cfg.reduce_chars - 2500
            for i, group in enumerate(groups):
                start = positions[group[0]["id"]]
                end = start + len(group)
                event(job, "reading", "正在阅读转写并保留主题细节", batch=i + 1, total=len(groups))
                allowed = {s["id"] for s in group}
                notes = client.structured(job, "reading", READ, {
                    "context": cfg.context, "segments": group,
                    "before": inputs[max(0, start - 2):start], "after": inputs[end:end + 2],
                    "output_char_limit": note_limit,
                }, ReadingNotes, lambda response, ids=allowed: validate_notes(response, ids, note_limit))
                for note in notes.topics:
                    materials.append({"id": f"n{len(materials) + 1:06d}", **note.model_dump()})
            if not materials:
                raise ValueError("转写中没有提取到实质主题；不会用背景知识填补缺失报告")
            plan = plan_topics(job, cfg, client, materials)
            validate_markdown(plan.introduction) if plan.introduction else None
            material_by_id = {note["id"]: note for note in materials}
            topic_titles = [topic.title for topic in plan.topics]
            body = [f"# {plan.title}"]
            if plan.introduction.strip():
                body.append(plan.introduction.strip())
            drafts = []
            for topic in plan.topics:
                markdown = write_topic(job, cfg, client, topic, material_by_id, by_id, topic_titles)
                body.append(markdown)
                drafts.append({"title": topic.title, "markdown": markdown})
            markdown = "\n\n".join(body).strip() + "\n"
            validate_markdown(markdown)
            result = {"key": key, "format": FORMAT_VERSION, "title": plan.title, "markdown": markdown,
                      "plan": plan.model_dump(), "topics": drafts, "materials": materials,
                      "input_segments": len(segments), "excluded_segments": excluded}
            write_json(completed, result)
            render_outputs(job, result)
        write_json(job / "report_status.json", {"status": "complete", "key": key})
        event(job, "complete", "主题总结已生成", report=str(job / "minutes.md"))
    except BaseException as e:
        write_json(job / "report_status.json", {"status": "failed", "key": key, "error": str(e)})
        raise
