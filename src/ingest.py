import pymupdf as fitz
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

from config import PDF_PATH, IMAGES_DIR
from embeddings import get_embed_model
from store import get_vector_store


def extract_text(pdf_path):
    """Un Document LlamaIndex par page de texte."""
    docs = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:
                docs.append(Document(
                    text=text,
                    metadata={"type": "text", "page": i, "source": pdf_path.name},
                ))
    return docs


def extract_tables(pdf_path):
    """Un Document par tableau, converti en Markdown (PyMuPDF)."""
    docs = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            for j, table in enumerate(page.find_tables().tables, start=1):
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows]
                md = "| " + " | ".join(rows[0]) + " |\n"
                md += "|" + "---|" * len(rows[0]) + "\n"
                for row in rows[1:]:
                    md += "| " + " | ".join(row) + " |\n"
                docs.append(Document(
                    text=md,
                    metadata={"type": "table", "page": i, "table_id": j, "source": pdf_path.name},
                ))
    return docs


def extract_images(pdf_path):
    """Sauvegarde les images bitmap dans images/ et retourne leurs infos."""
    images = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf, start=1):
            for j, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(pdf, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                path = IMAGES_DIR / f"page{i}_img{j}.png"
                pix.save(path)
                images.append({"path": str(path), "page": i, "source": pdf_path.name})
    return images


def build_index(pdf_path=PDF_PATH):
    Settings.embed_model = get_embed_model()
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    docs = extract_text(pdf_path) + extract_tables(pdf_path)
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True
    )
    print(f"{len(docs)} documents indexés dans PGVector")
    return index


if __name__ == "__main__":
    build_index()