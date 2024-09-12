import os
from dataclasses import dataclass

import vertexai
from google.oauth2 import service_account
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.vertex import Vertex
import logging
import sys

from ..library_custom.llama_index.Embedddings import VertexIEmbeddings

logger = logging.getLogger()


@dataclass
class _Config:

    _llm: LLM = None
    _embed_model: BaseEmbedding = None
    _context_window: int = 1000000
    _text_splitter = None
    _log_level: int = logging.INFO
    _base_dir: str = None
    _log_dir: str = 'logs'
    _log_file_name: str = 'error.log'

    def __init__(self):
        # Configurar el logger
        self._setup_logger()

        # Inicializar las credenciales de GCP
        credentials_path = self._get_env_variable('GCP_CREDENTIALS_PATH')
        if credentials_path is None:
            raise ValueError("No se ha definido la variable de entorno 'GCP_CREDENTIALS_PATH'.")
        credentials: service_account.Credentials = (
            service_account.Credentials.from_service_account_file(credentials_path)
        )
        vertexai.init(project=credentials.project_id, location='us-central1', credentials=credentials)
        logger.info("Credenciales de GCP inicializadas.")

        # Inicializar el modelo configuracion de Llama-Index
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.context_window = self.context_window
        Settings.text_splitter = self.text_splitter

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
        logger.info(f"LLM configurado: {llm.model}")

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
        logger.info(f"Modelo de Embeddings configurado: {embed_model.model_name}")

    def set_gcp_embed_model_name(self, embed_model_name: str):
        self.embed_model = VertexIEmbeddings(model_name=embed_model_name)

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

    # ---- Text Splitter ----
    @property
    def text_splitter(self):
        """Get the text splitter."""
        if self._text_splitter is None:
            self.text_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                embed_model=self.embed_model,
                include_metadata=True,
                include_prev_next_rel=True
            )
        return self._text_splitter

    @text_splitter.setter
    def text_splitter(self, text_splitter):
        """Set the text splitter."""
        self._text_splitter = text_splitter
        Settings.text_splitter = text_splitter

    # ---- Base Directory ----
    @property
    def base_dir(self) -> str:
        """Get the base directory."""
        if self._base_dir is None:
            self._base_dir = self._get_env_variable('SRC_DATA_PATH')
        os.makedirs(self._base_dir, exist_ok=True)
        return self._base_dir

    # ---- Log Directory ----
    @property
    def log_dir(self) -> str:
        """Get the log directory."""
        path = os.path.join(self.base_dir, self._log_dir)
        os.makedirs(path, exist_ok=True)
        return path


    ########################################################

    # ---- Get Enviroment Variable ----
    def _get_env_variable(self, var_name):
        """Obtiene una variable de entorno y emite una advertencia si no está definida."""
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Variable de entorno '{var_name}' no está definida.")
        logger.info(f"Variable de entorno '{var_name}' recuperada.")
        return value

    def _setup_logger(self):
        """Configura el logger."""
        # Configurar el nivel de logging del logger raíz
        logger.setLevel(self._log_level)

        # Crear handlers para registrar en la salida estándar y en un archivo
        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        errHandler = logging.FileHandler(os.path.join(self.log_dir, self._log_file_name))

        # Establecer los niveles de log en los handlers
        stdoutHandler.setLevel(logging.DEBUG)
        errHandler.setLevel(logging.ERROR)

        # Crear un formato de log usando atributos de Log Record
        fmt = logging.Formatter(
            "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
        )

        # Establecer el formato de log en cada handler
        stdoutHandler.setFormatter(fmt)
        errHandler.setFormatter(fmt)

        # Añadir cada handler al objeto Logger
        logger.addHandler(stdoutHandler)
        logger.addHandler(errHandler)

        return logger


# Configuracion
# Config = _Config()
