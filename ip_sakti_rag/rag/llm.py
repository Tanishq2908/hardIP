"""Thin Groq wrapper: retries on rate limits, JSON mode with a safe fallback,
support for reasoning models (openai/gpt-oss-*), and automatic fallback when a
configured model has been retired by Groq."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

log = logging.getLogger(__name__)
RETRYABLE = ("429", "rate limit", "rate_limit", "503", "502", "timeout", "overloaded", "temporarily")

# Preferred replacements if a configured model is not available on the account
MAIN_PREFERENCE = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
FAST_PREFERENCE = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def parse_json(text: str, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return default


def is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model or "qwen3" in model


class LLM:
    def __init__(self, api_key: str, model: str, fast_model: str):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model
        self.fast_model = fast_model
        self._checked = False

    # --------------------------------------------------------------- models
    def available_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self.client.models.list().data)
        except Exception as exc:
            log.warning("Could not list Groq models: %s", exc)
            return []

    def _resolve_models(self) -> None:
        """Swap out retired model IDs for working ones, once per process."""
        if self._checked:
            return
        self._checked = True
        avail = self.available_models()
        if not avail:
            return
        for attr, prefs in (("model", MAIN_PREFERENCE), ("fast_model", FAST_PREFERENCE)):
            current = getattr(self, attr)
            if current not in avail:
                repl = next((p for p in prefs if p in avail), None)
                if repl:
                    log.warning("Groq model %s is not available; using %s instead. "
                                "Update GROQ_%s in .env.", current, repl,
                                "MODEL" if attr == "model" else "FAST_MODEL")
                    setattr(self, attr, repl)

    # ----------------------------------------------------------------- chat
    def chat(self, messages: list[dict], *, fast: bool = False, temperature: float = 0.1,
             max_tokens: int = 1024, json_mode: bool = False, retries: int = 4) -> str:
        self._resolve_models()
        model = self.fast_model if fast else self.model
        kwargs: dict[str, Any] = dict(model=model, messages=messages, temperature=temperature)
        if is_reasoning_model(model):
            # Reasoning tokens count toward the limit, so leave room for them.
            kwargs["max_tokens"] = max_tokens + 2048
            kwargs["reasoning_effort"] = "low" if fast else "medium"
        else:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:
                msg = str(exc).lower()
                if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort")          # model does not accept it
                    continue
                if "model_not_found" in msg or "does not exist" in msg:
                    self._checked = False
                    self._resolve_models()
                    new_model = self.fast_model if fast else self.model
                    if new_model != kwargs["model"]:
                        kwargs["model"] = new_model
                        continue
                    raise RuntimeError(
                        f"Groq model '{model}' is not available. Run: python check_groq.py") from exc
                if attempt < retries - 1 and any(k in msg for k in RETRYABLE):
                    wait = 1.5 * (2 ** attempt)
                    log.warning("Groq busy (%s). Retrying in %.1fs", exc.__class__.__name__, wait)
                    time.sleep(wait)
                    continue
                raise
        return ""

    def json(self, system: str, user: str, *, fast: bool = True, max_tokens: int = 1024,
             default: Any = None) -> Any:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = self.chat(messages, fast=fast, json_mode=True, max_tokens=max_tokens, temperature=0)
        except Exception as exc:
            # JSON mode can fail validation on some outputs; retry once without it.
            if "json" not in str(exc).lower():
                raise
            text = self.chat(messages, fast=fast, max_tokens=max_tokens, temperature=0)
        out = parse_json(text, default)
        return default if out is None else out
