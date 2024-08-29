from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


_llm = OpenAI()
_embed_model = OpenAIEmbedding()

service_context_openai = ServiceContext.from_defaults(
    llm=_llm,
    embed_model=_embed_model,
    # node_parser=SentenceSplitter(chunk_size=512, chunk_overlap=20),
    # num_output=512,
    # context_window=3900,
)
