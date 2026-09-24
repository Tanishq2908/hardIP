"""Language detection and translation.

Translation goes through a Translator interface so Bhashini can replace Groq
without touching the pipeline. Legal names, section numbers and [n] citation
markers are always preserved.
"""
from __future__ import annotations

import os

from .metadata import detect_language

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "gu": "Gujarati", "mr": "Marathi", "ta": "Tamil",
    "bn": "Bengali", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia",
}


def detect_query_language(text: str, hint: str | None = None) -> str:
    """The question's own script decides the language.
    Devanagari is reported as Hindi unless the client says it is Marathi."""
    detected = detect_language(text)
    if detected == "unknown":
        detected = "en"
    if hint == "mr" and detected == "hi":
        return "mr"
    return detected


class Translator:
    def to_english(self, text: str, source: str) -> str:
        raise NotImplementedError

    def from_english(self, text: str, target: str) -> str:
        raise NotImplementedError


class GroqTranslator(Translator):
    SYSTEM = ("You are a precise legal translator. Translate the user's text from {src} to {tgt}. "
              "Keep Act names, section/article/rule numbers, organisation acronyms (NBA, SBB, FSSAI, "
              "WIPO, TKDL, CDSCO) and bracketed citation markers like [1] exactly as they are. "
              "Use simple, everyday words. Output only the translation.")

    def __init__(self, llm):
        self.llm = llm

    def _run(self, text: str, src: str, tgt: str) -> str:
        if not text.strip() or src == tgt:
            return text
        return self.llm.chat(
            [{"role": "system", "content": self.SYSTEM.format(src=LANG_NAMES.get(src, src),
                                                              tgt=LANG_NAMES.get(tgt, tgt))},
             {"role": "user", "content": text}],
            fast=False, temperature=0, max_tokens=2048).strip()

    def to_english(self, text: str, source: str) -> str:
        return self._run(text, source, "en")

    def from_english(self, text: str, target: str) -> str:
        return self._run(text, "en", target)


class BhashiniTranslator(Translator):
    """Placeholder for the Bhashini (ULCA) pipeline API.

    Fill in the two calls below with your Bhashini credentials
    (BHASHINI_USER_ID, BHASHINI_API_KEY) following the Bhashini pipeline docs.
    Until then the pipeline uses GroqTranslator.
    """

    def __init__(self):
        self.user_id = os.getenv("BHASHINI_USER_ID", "")
        self.api_key = os.getenv("BHASHINI_API_KEY", "")
        if not (self.user_id and self.api_key):
            raise RuntimeError("Bhashini credentials not set")

    def to_english(self, text: str, source: str) -> str:
        raise NotImplementedError("Implement Bhashini translation call here")

    def from_english(self, text: str, target: str) -> str:
        raise NotImplementedError("Implement Bhashini translation call here")


def make_translator(llm) -> Translator:
    if os.getenv("TRANSLATOR", "groq").lower() == "bhashini":
        return BhashiniTranslator()
    return GroqTranslator(llm)
