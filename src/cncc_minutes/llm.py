import json
import random
import threading
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

from .storage import digest, event, read_json, write_json


class LLMClient:
    """Shared across all video threads: global concurrency and start-rate limits."""

    def __init__(self, cfg, transport=None, sleep=time.sleep):
        self.cfg = cfg
        self.http = httpx.Client(timeout=httpx.Timeout(cfg.llm_timeout, connect=30), transport=transport)
        self.slots = threading.BoundedSemaphore(cfg.llm_concurrency)
        self.rate_lock = threading.Lock()
        self.next_start = 0.0
        self.sleep = sleep

    def close(self):
        self.http.close()

    def wait_rate(self):
        if not self.cfg.llm_rpm:
            return
        with self.rate_lock:
            now = time.monotonic()
            delay = max(0, self.next_start - now)
            self.next_start = max(now, self.next_start) + 60 / self.cfg.llm_rpm
        if delay:
            self.sleep(delay)

    def structured(self, job: Path, stage: str, system: str, payload, schema: type[BaseModel], validate=None):
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": system + "\n只返回 JSON 对象，必须符合此 schema：\n" + schema_text},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        def parse(value):
            result = schema.model_validate(json.loads(self.strip_fence(value)))
            if validate:
                validate(result)
            return result

        return self._complete(job, stage, messages, parse, "json")

    @staticmethod
    def strip_fence(text):
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        return text

    def markdown(self, job: Path, stage: str, system: str, payload, validate=None):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

        def parse(value):
            result = self.strip_fence(value)
            if not result:
                raise ValueError("Markdown 正文为空")
            if validate:
                validate(result)
            return result

        return self._complete(job, stage, messages, parse, "markdown")

    def _complete(self, job, stage, messages, parse, mode):
        key = digest({"identity": self.cfg.llm_identity(), "messages": messages, "mode": mode, "client_version": 2})
        path = job / "cache" / "llm" / f"{key}.json"
        original_messages = messages
        if path.exists():
            return parse(read_json(path)["content"])
        error = ""
        for attempt in range(self.cfg.llm_retries):
            retry_after = 0.0
            body = {"model": self.cfg.llm_model, "messages": messages,
                    "temperature": self.cfg.llm_temperature, "max_tokens": self.cfg.llm_max_tokens,
                    "stream": False, **self.cfg.llm_extra}
            if mode == "json" and self.cfg.llm_format == "json_object":
                body["response_format"] = {"type": "json_object"}
            try:
                with self.slots:
                    self.wait_rate()
                    response = self.http.post(
                        self.cfg.llm_base_url + "/chat/completions", json=body,
                        headers={"Authorization": f"Bearer {self.cfg.llm_api_key}"},
                    )
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    try:
                        retry_after = min(60.0, max(0.0, float(response.headers.get("retry-after", "0"))))
                    except ValueError:
                        pass
                    raise ValueError(f"LLM 服务暂时不可用，HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"LLM HTTP {response.status_code}：检查地址、密钥、模型名和 response_format / extra_body 配置。"
                    )
                data = response.json()
                choice = data["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"输出未正常结束: {choice.get('finish_reason')}，请增大 LLM_MAX_TOKENS 或缩小分段预算")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("LLM 没有返回正文（请检查思考模式及输出 token 限制）")
                result = parse(content)
                write_json(path, {"stage": stage, "model": self.cfg.llm_model,
                                  "content": content, "usage": data.get("usage", {})})
                event(job, stage, "完成 LLM 请求", attempt=attempt + 1, usage=data.get("usage", {}))
                return result
            except RuntimeError:
                raise
            except (ValueError, KeyError, IndexError, TypeError, httpx.TransportError) as e:
                error = str(e)[:1000] if not isinstance(e, httpx.TransportError) else type(e).__name__
                event(job, stage, "LLM 响应失败，保留此前缓存", attempt=attempt + 1, error=error)
                if attempt + 1 < self.cfg.llm_retries:
                    self.sleep(max(retry_after, min(30, 2 ** attempt) + random.uniform(0, 0.5)))
                # Schema errors get one bounded clarification; do not keep growing the context on retries.
                output = "JSON" if mode == "json" else "Markdown"
                messages = original_messages + [{"role": "user", "content":
                    f"上一次响应未通过校验，请重新生成完整 {output}。错误：" + error[:500]}]
        raise RuntimeError(f"LLM 请求经过 {self.cfg.llm_retries} 次仍失败：{error}")
