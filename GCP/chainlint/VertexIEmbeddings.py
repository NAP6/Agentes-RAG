from typing import (
    Any,
    List,
    Optional,
)

import vertexai
from google.oauth2 import service_account

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from llama_index.core.embeddings import BaseEmbedding

Embedding = List[float]

class VertexIEmbeddings(BaseEmbedding):

    _model: TextEmbeddingModel = None

    def __init__(
            self,
            model_name: str = "text-embedding-004",
            credentials_path: Optional[str] = None,
            **kwargs: Any,
    ) -> None:

        if credentials_path:
            credentials: service_account.Credentials = (
                service_account.Credentials.from_service_account_file(credentials_path)
            )
            vertexai.init(project=credentials.project_id, location='us-central1', credentials=credentials)

        if model_name:
            kwargs["model_name"] = model_name
            kwargs["_model"] = TextEmbeddingModel.from_pretrained(model_name)

        super().__init__(**kwargs)
        self._model = kwargs["_model"]

    def _get_text_embedding(self, text: str) -> Embedding:
        input_data = TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")
        embeddings = self._model.get_embeddings([input_data])
        return embeddings[0].values

    def _get_query_embedding(self, query: str) -> Embedding:
        input_data = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embeddings = self._model.get_embeddings([input_data])
        return embeddings[0].values

    async def _aget_query_embedding(self, query: str) -> Embedding:
        input_data = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embeddings = self._model.get_embeddings([input_data])
        return embeddings[0].values




    # async def _aget_text_embedding(self, text: str) -> Embedding:
    #     None
    #
    # def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
    #     None
    #
    # async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
    #     None