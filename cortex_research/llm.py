"""LLM providers behind one interface. With no key configured the system still works in
deterministic *extractive* mode (every sentence in the report is a verbatim source quote)."""
from __future__ import annotations
import json
import re
import httpx
from .config import settings


class LLM:
    name = "offline-extractive"
    available = False

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        raise NotImplementedError

    def json(self, system: str, user: str, max_tokens: int = 3000):
        raw = self.complete(system + "\nRespond with a single valid JSON object only.", user, max_tokens)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("LLM returned no JSON")
        return json.loads(m.group(0))


class OfflineLLM(LLM):
    pass


class AnthropicLLM(LLM):
    available = True

    def __init__(self, key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=60)
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system, user, max_tokens=2000):
        r = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class OpenAILLM(LLM):
    available = True

    def __init__(self, key: str, base: str, model: str):
        self.key, self.base, self.model = key, base.rstrip("/"), model
        self.name = f"openai:{model}"

    def complete(self, system, user, max_tokens=2000):
        r = httpx.post(f"{self.base}/chat/completions", timeout=60,
                       headers={"Authorization": f"Bearer {self.key}"},
                       json={"model": self.model, "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def get_llm() -> LLM:
    s = settings()
    if s.anthropic_key:
        return AnthropicLLM(s.anthropic_key, s.anthropic_model)
    if s.openai_key:
        return OpenAILLM(s.openai_key, s.openai_base, s.openai_model)
    return OfflineLLM()
