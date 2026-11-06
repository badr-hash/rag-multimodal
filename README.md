# 🧠 RAG Multimodal — Attention Is All You Need

RAG sur un PDF : texte + figures. Le projet fonctionne avec Postgres + pgvector, avec un repli local en mémoire quand la base ou le service Ollama n'est pas disponible.

## 🏗️ Architecture
- Texte → chunks → embedding BAAI/bge-small-en-v1.5
- Images → extraites du PDF → CLIP ViT-B/32
- Requête → index vectoriel local ou Postgres → réponse générée par Ollama ou un fallback local

## 🚀 Lancement
```bash
# 1. Mettre le PDF dans data/
cp "Attention is all you need.pdf" data/

# 2. Services optionnels
# docker compose up -d
# docker exec rag-ollama ollama pull llava

# 3. Python
python -m venv .venv
. .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 4. UI
streamlit run app.py
```

## ⚙️ Variables d'environnement
`PGVECTOR_HOST` · `PGVECTOR_PORT` · `PGVECTOR_DB` · `PGVECTOR_USER` · `PGVECTOR_PASSWORD` · `OLLAMA_URL` · `LLM_MODEL` · `TEXT_MODEL` · `CLIP_MODEL`

## 📁 Structure
```text
rag-multimodal/
├── data/
├── images/
├── src/
│   ├── config.py
│   ├── embeddings.py
│   ├── store.py
│   ├── ingest.py
│   ├── llm.py
│   └── rag.py
├── docker-compose.yml
├── requirements.txt
├── app.py
└── README.md
```