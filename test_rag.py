from dotenv import load_dotenv
load_dotenv()

from src.pipeline.database import get_connection, initialize_database, build_joined_view
from src.rag.embeddings import index_all_documents
from src.rag.retriever import retrieve, format_context

con = get_connection()
initialize_database(con)
build_joined_view(con)

# Build and index all documents
print("Indexing documents into ChromaDB...")
count = index_all_documents(con)
print(f"Total documents indexed: {count}\n")

# Test some searches
queries = [
    "How has UNH been performing lately?",
    "What is the current medical CPI trend?",
    "Which stocks have high volatility right now?",
]

for query in queries:
    print(f"Query: '{query}'")
    docs = retrieve(query, k=2)
    for doc in docs:
        print(f"  [{doc['score']}] {doc['text'][:120]}...")
    print()