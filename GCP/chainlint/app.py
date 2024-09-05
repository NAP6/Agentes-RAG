import chainlit as cl

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode

from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager

from llama_index.llms.vertex import Vertex
from VertexIEmbeddings import VertexIEmbeddings

credentials_path = 'C:\\Agentes-RAG\\GCP-Credentials\\llms-433815-5e7ca2a0c045.json'
Settings.embed_model = VertexIEmbeddings(credentials_path=credentials_path)
Settings.llm = Vertex(model="gemini-1.5-pro-001")
Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])
# Settings.context_window = 4096

storage_context = StorageContext.from_defaults(persist_dir="../Procesar Bloques Unstructured/data/index/vector_index_prueba")
index = load_index_from_storage(storage_context)

@cl.on_chat_start
async def start():
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.REACT, similarity_top_k=4, verbose=False)
    cl.user_session.set("chat_engine", chat_engine)

    await cl.Message(
        author="Assistant", content="Hola, soy un asistente IA como pueo ayudarte hoy?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="Assistant")
    await msg.send()

    res = chat_engine.chat(message.content)
    msg.content = res.response
    await msg.update()


    # res = chat_engine.stream_chat(message.content)
    #
    # for token in res.response_gen:
    #     await msg.stream_token(token.response)
    #
    # await msg.update()
