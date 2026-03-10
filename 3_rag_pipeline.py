# 3_rag_pipeline.py  ─── исправленный ───────────────────────────────────────────────

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

PERSIST_PATH = "./chroma_db"

# Настройки
Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=180.0, temperature=0.15)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Загрузка Chroma
db = chromadb.PersistentClient(path=PERSIST_PATH)
chroma_collection = db.get_collection("industrial_docs")          # уже должна существовать
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Загрузка storage context (теперь с persist_dir)
storage_context = StorageContext.from_defaults(
    persist_dir=PERSIST_PATH,          # ← ключевой момент
    vector_store=vector_store
)

# Загрузка индекса
index = load_index_from_storage(storage_context)

# Retriever + reranker
retriever = index.as_retriever(similarity_top_k=12)
# reranker = SentenceTransformerRerank(
#     model="BAAI/bge-reranker-v2-m3",   # multilingual, включая русский, топ-выбор 2025–2026
#     top_n=4
# )

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    # node_postprocessors=[reranker],
)

# Тестовый цикл
if __name__ == "__main__":
    print("Индекс загружен. Задавайте вопросы (exit для выхода)")
    while True:
        q = input("Вопрос: ").strip()
        if q.lower() in ["exit", "q", "выход", ""]:
            break
        try:
            response = query_engine.query(q)
            print("\nОтвет:", response.response.strip())
            print("\nИсточники:")
            for n in response.source_nodes:
                src = n.node.metadata.get("source", "unknown")
                print(f"  • {src[:90]}... (score: {n.score:.3f})")
        except Exception as e:
            print("Ошибка:", str(e))