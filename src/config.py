import os
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", BASE_DIR / "images"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PGVECTOR_HOST = os.getenv("PGVECTOR_HOST", "localhost")
PGVECTOR_PORT = int(os.getenv("PGVECTOR_PORT", "5432"))
PGVECTOR_DB = os.getenv("PGVECTOR_DB", "ragdb")
PGVECTOR_USER = os.getenv("PGVECTOR_USER", "rag")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD", "rag")
USE_PGVECTOR = _as_bool(os.getenv("USE_PGVECTOR"), default=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
VISION_MODEL = os.getenv("VISION_MODEL", "llava")
TEXT_MODEL = os.getenv("TEXT_MODEL", "BAAI/bge-small-en-v1.5")
CLIP_MODEL = os.getenv("CLIP_MODEL", "clip-ViT-B-32")
USE_MOCK_LLM = _as_bool(os.getenv("USE_MOCK_LLM"), default=False)

# Premier PDF trouvé dans data/ (pour les scripts lancés en ligne de commande)
_pdfs = sorted(DATA_DIR.glob("*.pdf"))
PDF_PATH = _pdfs[0] if _pdfs else DATA_DIR / "document.pdf"