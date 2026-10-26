import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def boolean(value: str) -> bool:
    if value.lower() not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"无效布尔值: {value}")
    return value.lower() in {"true", "1", "yes"}


@dataclass
class Settings:
    root: Path
    input_dir: Path
    output_dir: Path
    asr_model: Path
    aligner_model: Path
    diarization_model: Path
    gpus: tuple[str, ...] = ("0", "1", "2", "3")
    language: str = "Chinese"
    batch_size: int = 2
    chunk_seconds: float = 25
    max_new_tokens: int = 512
    attention: str = "sdpa"
    word_timestamps: bool = True
    asr_use_glossary: bool = False
    diarization: bool = False
    num_speakers: int = 0
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_concurrency: int = 4
    llm_rpm: int = 0
    llm_timeout: float = 180
    llm_retries: int = 4
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.1
    llm_format: str = "json_object"
    llm_extra: dict = field(default_factory=dict)
    summary_chars: int = 10000
    reduce_chars: int = 16000
    allow_knowledge: bool = True
    summary_style: str = ""
    glossary: str = ""
    speaker_names: dict = field(default_factory=dict)
    context: str = "中文技术论坛"
    ffmpeg_path: str = ""

    @classmethod
    def load(cls, env_path: Path) -> "Settings":
        env_path = env_path.resolve()
        load_dotenv(env_path, override=False)
        root = env_path.parent

        def path(key, default):
            p = Path(os.getenv(key, default)).expanduser()
            return (p if p.is_absolute() else root / p).resolve()

        glossary_path = path("GLOSSARY_PATH", "glossary.txt")
        names_path = path("SPEAKER_NAMES_PATH", "speaker_names.json")
        glossary = "\n".join(
            line.strip() for line in (glossary_path.read_text("utf-8").splitlines() if glossary_path.exists() else [])
            if line.strip() and not line.lstrip().startswith("#")
        )
        names = json.loads(names_path.read_text("utf-8")) if names_path.exists() else {}
        if not isinstance(names, dict) or any(not isinstance(v, dict) for v in names.values()):
            raise ValueError("speaker_names.json 应为 {视频文件名: {SPEAKER_00: 姓名}}")
        s = cls(
            root=root, input_dir=path("INPUT_DIR", "video"), output_dir=path("OUTPUT_DIR", "output"),
            asr_model=path("ASR_MODEL_PATH", "models/Qwen3-ASR-1.7B"),
            aligner_model=path("ALIGNER_MODEL_PATH", "models/Qwen3-ForcedAligner-0.6B"),
            diarization_model=path("DIARIZATION_MODEL_PATH", "models/speaker-diarization-community-1"),
            gpus=tuple(x.strip() for x in os.getenv("GPU_DEVICES", "0,1,2,3").split(",") if x.strip()),
            language=os.getenv("ASR_LANGUAGE", "Chinese"),
            batch_size=int(os.getenv("ASR_BATCH_SIZE", "2")),
            chunk_seconds=float(os.getenv("ASR_CHUNK_SECONDS", "25")),
            max_new_tokens=int(os.getenv("ASR_MAX_NEW_TOKENS", "512")),
            attention=os.getenv("ASR_ATTENTION", "sdpa"),
            word_timestamps=boolean(os.getenv("ASR_WORD_TIMESTAMPS", "true")),
            asr_use_glossary=boolean(os.getenv("ASR_USE_GLOSSARY", "false")),
            diarization=boolean(os.getenv("DIARIZATION", "false")),
            num_speakers=int(os.getenv("DIARIZATION_NUM_SPEAKERS", "0")),
            llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", ""), llm_model=os.getenv("LLM_MODEL", ""),
            llm_concurrency=int(os.getenv("LLM_CONCURRENCY", "4")),
            llm_rpm=int(os.getenv("LLM_REQUESTS_PER_MINUTE", "0")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
            llm_retries=int(os.getenv("LLM_RETRIES", "4")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            llm_format=os.getenv("LLM_RESPONSE_FORMAT", "json_object"),
            llm_extra=json.loads(os.getenv("LLM_EXTRA_BODY", "{}")),
            summary_chars=int(os.getenv("SUMMARY_CHARS", "10000")),
            reduce_chars=int(os.getenv("REDUCE_INPUT_CHARS", "16000")),
            allow_knowledge=boolean(os.getenv("SUMMARY_ALLOW_KNOWLEDGE", "true")),
            summary_style=os.getenv("SUMMARY_STYLE", ""),
            glossary=glossary, speaker_names=names, context=os.getenv("MEETING_CONTEXT", "中文技术论坛"),
            ffmpeg_path=os.getenv("FFMPEG_PATH", ""),
        )
        if not s.gpus or len(set(s.gpus)) != len(s.gpus):
            raise ValueError("GPU_DEVICES 不能为空或重复")
        if any(not x.isdigit() for x in s.gpus):
            raise ValueError("GPU_DEVICES 使用本机可见 GPU 的数字编号")
        if not 5 <= s.chunk_seconds <= 30:
            raise ValueError("ASR_CHUNK_SECONDS 应在 5–30 秒之间")
        for key in ("batch_size", "max_new_tokens", "llm_concurrency", "llm_timeout", "llm_retries", "llm_max_tokens"):
            if getattr(s, key) <= 0:
                raise ValueError(f"{key} 必须大于 0")
        for key in ("summary_chars", "reduce_chars"):
            if getattr(s, key) < 1000:
                raise ValueError(f"{key} 必须至少为 1000")
        if s.reduce_chars < 4200:
            raise ValueError("REDUCE_INPUT_CHARS 至少需要 4200，建议 16000")
        if s.llm_rpm < 0 or s.num_speakers < 0:
            raise ValueError("RPM / 发言人数不得为负数")
        if s.llm_format not in {"json_object", "none"}:
            raise ValueError("LLM_RESPONSE_FORMAT 只能是 json_object 或 none")
        if not isinstance(s.llm_extra, dict):
            raise ValueError("LLM_EXTRA_BODY 必须是 JSON 对象")
        if set(s.llm_extra) & {"model", "messages", "stream", "response_format", "max_tokens", "temperature"}:
            raise ValueError("LLM_EXTRA_BODY 不可覆盖核心请求字段")
        if len(s.glossary) > 12000:
            raise ValueError("术语表超过 12000 字符，请精简后使用")
        return s

    def validate_llm(self):
        if not self.llm_base_url.startswith(("http://", "https://")) or ".example" in self.llm_base_url:
            raise ValueError("请在 .env 中配置真实 LLM_BASE_URL")
        if not self.llm_api_key or self.llm_api_key.startswith("replace-"):
            raise ValueError("请配置 LLM_API_KEY；本地免鉴权服务可填 local")
        if not self.llm_model or self.llm_model.startswith("replace-"):
            raise ValueError("请配置 LLM_MODEL")

    def llm_identity(self):
        return {"url": self.llm_base_url, "model": self.llm_model, "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens, "format": self.llm_format, "extra": self.llm_extra}
