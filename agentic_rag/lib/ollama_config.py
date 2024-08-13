from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.langchain import LangchainEmbedding
from lib.OllamaEmbeddings import OllamaEmbeddings


_llm = Ollama(base_url="http://192.168.1.174:11434/", model='llama3.1')
_embeddings = OllamaEmbeddings(model='llama3.1', base_url="http://192.168.1.174:11434")
_embed_model = LangchainEmbedding(_embeddings)

service_context_ollama = ServiceContext.from_defaults(
    llm=_llm,
    embed_model=_embed_model,
    # node_parser=SentenceSplitter(chunk_size=512, chunk_overlap=20),
    # num_output=512,
    # context_window=3900,
)