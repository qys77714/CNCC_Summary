import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text("utf-8"))


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_signature(path: Path):
    # Full streaming hash: detects in-place replacement and remains valid after copying to a different host.
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return {"sha256": h.hexdigest(), "size": path.stat().st_size}


def model_signature(path: Path):
    # No multi-GB weight hashing on every video. Include every file's metadata and small configuration contents.
    entries = []
    if not path.is_dir():
        return {"path": str(path), "missing": True}
    for p in sorted(path.rglob("*")):
        if p.is_file() and ".cache" not in p.relative_to(path).parts:
            stat = p.stat()
            small_hash = hashlib.sha256(p.read_bytes()).hexdigest() if stat.st_size < 65536 else None
            entries.append((str(p.relative_to(path)), stat.st_size, stat.st_mtime_ns, small_hash))
    return digest(entries)


def job_id(source: Path, input_root: Path) -> str:
    try:
        relative = str(source.relative_to(input_root))
    except ValueError:
        relative = source.name
    name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", source.stem).strip("_")[:64]
    return f"{name}-{digest(relative)[:10]}"


def timestamp(seconds: float, milliseconds=False) -> str:
    ms = max(0, round(seconds * 1000))
    sec, ms = divmod(ms, 1000)
    minute, sec = divmod(sec, 60)
    hour, minute = divmod(minute, 60)
    text = f"{hour:02d}:{minute:02d}:{sec:02d}"
    return text + f",{ms:03d}" if milliseconds else text


def event(job: Path, stage: str, message: str, **extra):
    job.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": stage, "message": message, **extra}
    # Each job is owned by one GPU worker and then one LLM worker; no concurrent writes per job.
    with (job / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(job / "status.json", record)


@contextlib.contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"{root} 已有任务在运行。请复用现有运行或使用另一 OUTPUT_DIR。") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
