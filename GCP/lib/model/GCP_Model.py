from typing import (
    Any,
    List,
    Optional,
    Union,
    Dict,
)

import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Image
)

from langchain_google_vertexai import VertexAI
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field
from langchain_core.callbacks import CallbackManagerForLLMRun


ContentDict = Dict[str, Any]
ContentsType = Union[
    List["Content"],
    List[ContentDict],
    str,
    Image,
    Part,
    List[Union[str, "Image", "Part"]],
]

class GCP_Model(LLM):

    model_name: str = Field(default="text-bison", alias="model")
    vertex_client: GenerativeModel = None

    def __init__(self,
                 model_name: str,
                 credentials_path: Optional[str] = None,
                 **kwargs: Any
                 ):
        if credentials_path:
            credentials: service_account.Credentials = (
                service_account.Credentials.from_service_account_file(credentials_path)
            )
            vertexai.init(project=credentials.project_id, location='us-central1', credentials=credentials)

            if model_name:
                kwargs["model_name"] = model_name
                kwargs["vertex_client"] =GenerativeModel(model_name)

            super().__init__(**kwargs)

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        text_part = Part.from_text(prompt)
        contents = [text_part]

        generated = self.vertex_client.generate_content(contents)
        generated_text = generated.candidates[0].content.parts[0].text
        
        return generated_text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Devuelve los parámetros de identificación del modelo.

        Returns:
            Un diccionario con los parámetros de identificación.
        """
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "vertexai"
