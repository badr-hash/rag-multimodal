import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
from config import DATA_DIR

st.set_page_config(page_title="RAG multimodal PDF", layout="wide")
st.title("RAG multimodal sur PDF")


@st.cache_resource
def load_rag():
    import rag
    return rag


# ---------- Sidebar : chargement du PDF ----------
with st.sidebar:
    st.header("1. Charger un PDF")
    uploaded = st.file_uploader("Choisir un PDF", type="pdf")

    if uploaded and st.button("Indexer ce PDF"):
        from reset import reset_tables
        from ingest import build_index
        from images import index_images

        pdf_path = DATA_DIR / uploaded.name
        pdf_path.write_bytes(uploaded.getbuffer())

        try:
            with st.spinner("Indexation en cours (texte, tableaux, images)..."):
                reset_tables()
                build_index(pdf_path)
                index_images(pdf_path)
            st.session_state["indexed"] = uploaded.name
            st.session_state["history"] = []
            st.success(f"{uploaded.name} indexé")
        except Exception as exc:
            st.error(f"L'indexation a échoué : {exc}")

    if "indexed" in st.session_state:
        st.info(f"Document actif : {st.session_state['indexed']}")
    else:
        st.caption("Si ton PDF est déjà indexé, tu peux poser des questions directement.")

# ---------- Zone principale : questions ----------
st.header("2. Poser une question")

if "history" not in st.session_state:
    st.session_state["history"] = []

for item in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander("Sources"):
            st.write(item["sources"])
        for path in item["images"]:
            st.image(path, width=400)

question = st.chat_input("Pose ta question sur le document")

if question:
    if "indexed" not in st.session_state:
        st.warning("Indexe d'abord un PDF pour lancer une recherche.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Recherche et génération..."):
                    rag = load_rag()
                    answer, sources = rag.ask(question)
                    image_paths = []
                    if any(t == "image" for t, _ in sources):
                        from images import search_images
                        image_paths = [p for p, _ in search_images(question, top_k=2)]
            except Exception as exc:
                answer = f"Je n'ai pas pu répondre à cette question dans l'environnement actuel. Détail : {exc}"
                sources = []
                image_paths = []
            st.write(answer)
            with st.expander("Sources"):
                st.write(sources)
            for path in image_paths:
                st.image(path, width=400)

        st.session_state["history"].append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "images": image_paths,
        })