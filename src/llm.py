import os

from llama_index.core.llms.mock import MockLLM

try:
    from llama_index.llms.ollama import Ollama
except Exception:  # pragma: no cover - optional dependency
    Ollama = None

from config import LLM_MODEL, OLLAMA_URL, USE_MOCK_LLM


def get_llm():
    if USE_MOCK_LLM:
        return MockLLM()
    if Ollama is None:
        return MockLLM()

    try:
        return Ollama(
            model=os.getenv("LLM_MODEL", LLM_MODEL),
            base_url=os.getenv("OLLAMA_URL", OLLAMA_URL),
            request_timeout=300.0,
            temperature=0.1,
        )
    except Exception:
        return MockLLM()