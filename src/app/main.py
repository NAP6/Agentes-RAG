from config import Config
from agent.load_index import NodeLoaderAgent

indx_agent = NodeLoaderAgent(
    pkl_path='C:/Agentes-RAG/GCP/Procesar Bloques Unstructured/embeded_nodes_v2.pkl',
    verbose=False
)