import os

import ollama
from PIL import Image
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery

from config import CLIP_MODEL, PDF_PATH
from embeddings import get_embed_model
from ingest import extract_images
from store import get_vector_store

try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception:  # pragma: no cover - optional dependency
    PGVectorStore = None

clip_model = SentenceTransformer(CLIP_MODEL)
_CLIP_STORE_CACHE = {}


def get_clip_store():
    if "clip_images" in _CLIP_STORE_CACHE:
        return _CLIP_STORE_CACHE["clip_images"]

    if PGVectorStore is None:
        store = SimpleVectorStore()
    else:
        try:
            store = PGVectorStore.from_params(
                host=os.getenv("PGVECTOR_HOST", "localhost"),
                port=int(os.getenv("PGVECTOR_PORT", "5432")),
                database=os.getenv("PGVECTOR_DB", "ragdb"),
                user=os.getenv("PGVECTOR_USER", "rag"),
                password=os.getenv("PGVECTOR_PASSWORD", "rag"),
                table_name="clip_images",
                embed_dim=512,
            )
        except Exception:
            store = SimpleVectorStore()

    _CLIP_STORE_CACHE["clip_images"] = store
    return store


def describe_image(path):
    """Description textuelle d'une image avec llava (Ollama)."""
    try:
        res = ollama.chat(
            model="llava",
            messages=[{
                "role": "user",
                "content": "This image comes from a technical document. Describe precisely what is drawn: boxes, arrows, labels, text visible, and what process or system it represents. Do not guess if unsure.",
                "images": [path],
            }],
        )
        return res["message"]["content"]
    except Exception:
        return "Image extracted from the PDF. The environment could not analyze this figure automatically."


def index_images(pdf_path=PDF_PATH):
    Settings.embed_model = get_embed_model()
    images = extract_images(pdf_path)

    docs = []
    for img in images:
        desc = describe_image(img["path"])
        print(f"[page {img['page']}] {desc[:120]}...")
        docs.append(Document(
            text=desc,
            metadata={
                "type": "image",
                "page": img["page"],
                "image_path": img["path"],
                "source": img["source"],
            },
        ))
    if docs:
        storage = StorageContext.from_defaults(vector_store=get_vector_store())
        VectorStoreIndex.from_documents(docs, storage_context=storage)

    nodes = []
    for img in images:
        try:
            emb = clip_model.encode(Image.open(img["path"])).tolist()
        except Exception:
            emb = [0.0] * 512
        nodes.append(TextNode(
            text=img["path"],
            embedding=emb,
            metadata={"type": "image", "page": img["page"], "image_path": img["path"]},
        ))
    if nodes:
        get_clip_store().add(nodes)
    print(f"{len(images)} images indexées (llava + CLIP)")


def search_images(query, top_k=2):
    """Recherche texte -> image avec CLIP."""
    try:
        q = clip_model.encode(query).tolist()
        res = get_clip_store().query(VectorStoreQuery(query_embedding=q, similarity_top_k=top_k))
        return [(n.metadata["image_path"], n.metadata["page"]) for n in res.nodes]
    except Exception:
        return []


if __name__ == "__main__":
    index_images()
    print(search_images("a diagram"))
