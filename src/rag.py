from llama_index.core import PromptTemplate, Settings, VectorStoreIndex

from embeddings import get_embed_model
from llm import get_llm
from store import get_vector_store

Settings.embed_model = get_embed_model()
Settings.llm = get_llm()

QA_PROMPT = PromptTemplate(
    "You are an assistant answering questions about a technical PDF document.\n"
    "Use ONLY the context below. If the answer is not in the context, say you cannot find it.\n"
    "Always mention the page number(s) you used.\n"
    "Answer in the same language as the question.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


def get_query_engine(top_k=4):
    try:
        store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(store)
        return index.as_query_engine(
            similarity_top_k=top_k,
            text_qa_template=QA_PROMPT,
        )
    except Exception:
        return None


def ask(question):
    engine = get_query_engine()
    if engine is None:
        return "Je n'ai pas trouvé de contenu indexé pour ce document. Veuillez d'abord indexer un PDF.", []

    response = engine.query(question)
    sources = [
        (n.metadata.get("type"), n.metadata.get("page"))
        for n in response.source_nodes
    ]
    return str(response), sources


if __name__ == "__main__":
    question = input("Question : ")
    answer, sources = ask(question)
    print("\nRéponse :\n", answer)
    print("\nSources (type, page) :", sources)
