from llama_index.core.vector_stores import SimpleVectorStore

from config import (
    PGVECTOR_DB,
    PGVECTOR_HOST,
    PGVECTOR_PASSWORD,
    PGVECTOR_PORT,
    PGVECTOR_USER,
    USE_PGVECTOR,
)

try:
    from llama_index.vector_stores.postgres import PGVectorStore
except Exception:  # pragma: no cover - optional dependency
    PGVectorStore = None

_VECTOR_STORE_CACHE = {}


def get_vector_store(table_name="documents"):
    if table_name in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[table_name]

    if not USE_PGVECTOR or PGVectorStore is None:
        store = SimpleVectorStore()
    else:
        try:
            store = PGVectorStore.from_params(
                host=PGVECTOR_HOST,
                port=PGVECTOR_PORT,
                database=PGVECTOR_DB,
                user=PGVECTOR_USER,
                password=PGVECTOR_PASSWORD,
                table_name=table_name,
                embed_dim=384,
            )
        except Exception:
            store = SimpleVectorStore()

    _VECTOR_STORE_CACHE[table_name] = store
    return store


def reset_vector_stores():
    _VECTOR_STORE_CACHE.clear()
