from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.objects import ObjectIndex

from config import Config
from agent.DocumentComponent import DocumentComponent

doc_voyager = DocumentComponent(
    pkl_path=r'C:\Agentes-RAG\src\data\nodes\VOYAGER An Open-Ended Embodied Agent.pkl',
    verbose=True
)

doc_rag = DocumentComponent(
    pkl_path=r'C:\Agentes-RAG\src\data\nodes\Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pkl',
    verbose=True
)

obj_index = ObjectIndex.from_objects(
    [doc_rag.document_tool, doc_voyager.document_tool],
    index_cls=VectorStoreIndex,
)

agent = ReActAgent.from_tools(
    tool_retriever=obj_index.as_retriever(),
    verbose=True
)