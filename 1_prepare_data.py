from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

documents = SimpleDirectoryReader("data", filename_as_id=True).load_data()

# Лучший chunking 2026 для техдокументации
splitter = SentenceSplitter(
    chunk_size=512,          # токенов
    chunk_overlap=100,       # 20% — золотой стандарт
    include_metadata=True
)
nodes = splitter.get_nodes_from_documents(documents)

# Добавляем метаданные (очень важно!)
for node in nodes:
    node.metadata["source"] = node.id_
    node.metadata["doc_type"] = "instruction" if "инструкция" in node.text.lower() else "report"