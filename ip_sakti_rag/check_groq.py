"""Check your Groq key and list the models your account can use.

  python check_groq.py
"""
import sys

from config import settings
from rag.llm import LLM

llm = LLM(settings.groq_api_key, settings.groq_model, settings.groq_fast_model)
models = llm.available_models()
if not models:
    sys.exit("Could not reach Groq. Check GROQ_API_KEY in .env and your internet connection.")
print("Models available to your key:")
for m in models:
    print("  ", m)
print(f"\nConfigured main model: {settings.groq_model}  {'OK' if settings.groq_model in models else 'NOT AVAILABLE'}")
print(f"Configured fast model: {settings.groq_fast_model}  {'OK' if settings.groq_fast_model in models else 'NOT AVAILABLE'}")
print("\nTest call...")
print("Reply:", llm.chat([{"role": "user", "content": "Reply with just the word: ready"}], fast=True, max_tokens=20))
