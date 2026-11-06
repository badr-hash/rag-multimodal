import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import TEXT_MODEL


def get_embed_model():
    return HuggingFaceEmbedding(model_name=os.getenv("TEXT_MODEL", TEXT_MODEL))