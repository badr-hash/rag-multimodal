from llama_index.core import Settings, VectorStoreIndex

from embeddings import get_embed_model
from store import get_vector_store

Settings.embed_model = get_embed_model()
index = VectorStoreIndex.from_vector_store(get_vector_store())
retriever = index.as_retriever(similarity_top_k=3)

for r in retriever.retrieve("What is the main idea of this document?"):
    print(f"[page {r.metadata.get('page')}] score={r.score:.3f}")
    print(r.text[:200].replace("\n", " "), "\n")