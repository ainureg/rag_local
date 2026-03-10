# 2_build_index.py  ─── исправленный и полный вариант ────────────────────────────────

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# 1. Загрузка и чанкинг
documents = SimpleDirectoryReader(
    input_dir="data",
    required_exts=[".pdf", ".docx", ".txt", ".md"],
).load_data()

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)

for node in nodes:
    node.metadata["source"] = node.node_id   # или node.ref_doc_id, как удобнее

# 2. Embedding модель
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# 3. Chroma
PERSIST_PATH = "./chroma_db"
db = chromadb.PersistentClient(path=PERSIST_PATH)
chroma_collection = db.get_or_create_collection("industrial_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Создаём индекс
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

# 5. ← Самое важное! Сохраняем ВСЁ на диск
index.storage_context.persist(persist_dir=PERSIST_PATH)

print(f"Индекс создан и сохранён. Чанков: {len(nodes)}")
print(f"Папка: {PERSIST_PATH}")