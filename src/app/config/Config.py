import os
from dataclasses import dataclass

import vertexai
from google.oauth2 import service_account
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.llms.vertex import Vertex

from ..lib.library_custom.llama_index.Embedddings import VertexIEmbeddings


@dataclass
class _Config:

    _gcp_credentials = None
    _llm: LLM = None
    _embed_model: BaseEmbedding = None
    _context_window: int = 1000000

    def __init__(self):
        credentials_path = os.getenv('GCP_CREDENTIALS_PATH')
        if credentials_path is None:
            raise ValueError("No se ha definido la variable de entorno 'GCP_CREDENTIALS_PATH'.")
        credentials: service_account.Credentials = (
            service_account.Credentials.from_service_account_file(credentials_path)
        )
        vertexai.init(project=credentials.project_id, location='us-central1', credentials=credentials)

        # Inicializar el modelo configuracion de Llama-Index
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.context_window = self.context_window

    # --- LLM ---
    @property
    def llm(self):
        """Getter para recuperar LLM."""
        if self._llm is None:
            self.set_gcp_llm_by_name("gemini-1.5-pro-001")
        return self._llm

    @llm.setter
    def llm(self, llm: LLM):
        """Setter para asignar el LLM."""
        self._llm = llm
        Settings.llm = llm

    def set_gcp_llm_by_name(self, llm_name: str):
        self.llm = Vertex(model=llm_name)


    # ---- Embedding ----
    @property
    def embed_model(self) -> BaseEmbedding:
        """Get the embedding model."""
        if self._embed_model is None:
            self.set_gcp_embed_model_name("text-embedding-004")
        return self._embed_model

    @embed_model.setter
    def embed_model(self, embed_model: BaseEmbedding):
        """Setter para asignar el modelo de Embeddings."""
        self._embed_model = embed_model
        Settings.embed_model = embed_model

    def set_gcp_embed_model_name(self, embed_model_name: str):
        self.embed_model = VertexIEmbeddings(model_name = embed_model_name)

    # ---- Context Window ----
    @property
    def context_window(self) -> int:
        """Get the context window."""
        return self._context_window

    @context_window.setter
    def context_window(self, context_window: int):
        """Set the context window."""
        self._context_window = context_window
        Settings.context_window = context_window




# def __init__(self):
#     # Configurar rutas de directorios
#     self.base_dir = self.get_env_variable('SRC_DATA_PATH', '/')
#     self.node_dir = os.path.join(self.base_dir, "nodes")
#     self.log_dir = os.path.join(self.base_dir, "logs")
#
#     # Configurar otras variables específicas
#     self.llm_model = "gemini-1.5-pro-001"
#     self.gcp_credentials_path = self.get_env_variable('GCP_CREDENTIALS_PATH', None)
#
#     # Verificar que se haya definido la variable de entorno 'GCP_CREDENTIALS_PATH'
#     if self.gcp_credentials_path is None:
#         print("Error: No se ha definido la variable de entorno 'GCP_CREDENTIALS_PATH'.")
#         raise ValueError("No se ha definido la variable de entorno 'GCP_CREDENTIALS_PATH'.")
#
#     # Asegurar la existencia de directorios
#     self.ensure_directories_exist()
#
#     # Imprimir configuración
#     print("Configuración:")
#     print(f"  node_dir: {self.node_dir}")
#     print(f"  log_dir: {self.log_dir}")
#     print(f"  llm_model: {self.llm_model}")
#
# def get_env_variable(self, var_name, default=None):
#     """Obtiene una variable de entorno y emite una advertencia si no está definida."""
#     value = os.getenv(var_name)
#     if value is None:
#         if default is None:
#             print(f"Advertencia: Variable de entorno '{var_name}' no está definida y no hay un valor por defecto.")
#         else:
#             print(f"Advertencia: Variable de entorno '{var_name}' no está definida. Usando valor por defecto: {default}")
#         return default
#     return value
#
# def ensure_directories_exist(self):
#     """Crea los directorios si no existen."""
#     if not os.path.exists(self.node_dir):
#         print(f'WARNING: El directorio {self.node_dir} no existe.')
#     os.makedirs(self.node_dir, exist_ok=True)
#     os.makedirs(self.log_dir, exist_ok=True)

Config = _Config()
