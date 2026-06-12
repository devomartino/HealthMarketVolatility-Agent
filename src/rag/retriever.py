import logging
from src.rag.embeddings import get_collection

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Search the vector store for documents most relevant to the query.
    Returns top-k results sorted by similarity.
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Vector store is empty. Run index_all_documents() first.")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "text": text,
            "source": meta.get("source", ""),
            "date": meta.get("date", ""),
            "score": round(1 - dist, 4),
        })

    return docs


def format_context(docs: list[dict]) -> str:
    """Format retrieved docs into a context block for Claude."""
    if not docs:
        return "No relevant context found in the knowledge base."

    lines = ["[Retrieved Context]"]
    for i, doc in enumerate(docs, 1):
        lines.append(
            f"\n--- Source {i} ({doc['source']}, {doc['date']}, "
            f"similarity={doc['score']}) ---"
        )
        lines.append(doc["text"])
    return "\n".join(lines)