import logging
import pickle
from dataclasses import dataclass
from typing import Optional, Sequence, List
import re

from llama_index.core import SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import BaseNode, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata

logger = logging.getLogger(__name__)
logger.propagate = True

@dataclass
class DocumentComponent:
    """
        # DocumentComponent

        ## Descripción
        Componente que representa un documento y sus herramientas de consulta.

        ## Atributos
        - **verbose** (*bool*): Indica si se deben mostrar mensajes de depuración.
        - **title_of_document** (*str*): Título del documento.
        - **summary_of_document** (*str*): Resumen del documento.
        - **vector_index** (*VectorStoreIndex*): Índice de vectores.
        - **summary_index** (*SummaryIndex*): Índice de resumen.

    """

    _verbose: bool = False
    _title_of_document: str = None
    _summary_of_document: str = None
    _vector_index: VectorStoreIndex = None
    _summary_index: SummaryIndex = None

    def __init__(self,
                 pkl_path: str,
                 verbose: Optional[bool] = False,
                 sumary_of_document: Optional[str] = None,
                 ):
        """
        Inicializa un nuevo DocumentComponent.
        :param pkl_path:
        :param verbose:
        """
        self.verbose = verbose
        nodes: Sequence[BaseNode] = self._load_nodes(pkl_path)
        self._title_of_document = nodes[0].metadata.get('title_of_the_document')
        self._vector_index = VectorStoreIndex(nodes)
        self._summary_index = SummaryIndex(nodes)
        if sumary_of_document is not None:
            self._summary_of_document = sumary_of_document


    # ---- Verbose ----
    @property
    def verbose(self) -> bool:
        """Retorna el valor de verbose."""
        if self._verbose is None:
            self._verbose = False
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        """Asigna el valor de verbose."""
        self._verbose = verbose

    # ---- Título del documento ----
    @property
    def title_of_document(self) -> str:
        """Retorna el título del documento."""
        return self._title_of_document

    # ---- Resumen del documento ----
    @property
    def summary_of_document(self) -> str:
        """Retorna el resumen del documento."""
        if self._summary_of_document is None:
            self._summary_of_document = self.summary_qe.query("Give me a summary").response
        return self._summary_of_document

    # ---- Índice de vectores ----
    @property
    def vector_index(self) -> VectorStoreIndex:
        """Retorna el índice de vectores."""
        return self._vector_index

    # ---- Índice de resumen ----
    @property
    def summary_index(self) -> SummaryIndex:
        """Retorna el índice de resumen."""
        return self._summary_index

    # ---- Query Engine de vectores ----
    @property
    def vector_qe(self) -> BaseQueryEngine:
        """Retorna el Query Engine de vectores."""
        return self.vector_index.as_query_engine()

    # ---- Query Engine de resumen ----
    @property
    def summary_qe(self) -> BaseQueryEngine:
        """Retorna el Query Engine de resumen."""
        return self.summary_index.as_query_engine()

    # ---- Herramienta de Vector Query Engine ----
    @property
    def vector_tool(self) -> QueryEngineTool:
        """Retorna la herramienta de Vector Query Engine."""

        sanitized_title = re.sub(r'[^a-zA-Z0-9]', '_', self.title_of_document.lower())  # Reemplaza caracteres no permitidos
        return QueryEngineTool(
            query_engine=self.vector_qe,
            metadata=ToolMetadata(
                name=f"vector_tool",
                description=(
                    "Useful for questions related to specific aspects of"
                    f" '{self.title_of_document}' (e.g. specific sections, key arguments,"
                    " research findings, literature reviews, or more)"
                ),
            ),
        )

    # ---- Herramienta de Summary Query Engine ----
    @property
    def summary_tool(self) -> QueryEngineTool:
        """Retorna la herramienta de Summary Query Engine."""
        sanitized_title = re.sub(r'[^a-zA-Z0-9]', '_', self.title_of_document.lower())  # Reemplaza caracteres no permitidos
        return QueryEngineTool(
            query_engine=self.summary_qe,
            metadata=ToolMetadata(
                name=f"summary_tool",
                description=(
                    "Useful for any requests that require a holistic summary"
                    f" of EVERYTHING about '{self.title_of_document}'. For questions about"
                    " more specific sections, please use the vector_tool."
                ),
            ),
        )

    # ---- Lista de herramientas de consulta ----
    @property
    def query_engine_tools(self) -> List[QueryEngineTool]:
        """Retorna la lista de herramientas de consulta."""
        return [self.vector_tool, self.summary_tool]

    # ---- ReAct Agent ----
    @property
    def agent_ReAct(self) -> ReActAgent:
        """Retorna el agente ReAct."""
        return ReActAgent.from_tools(
            self.query_engine_tools,
            verbose=self.verbose,
            context=(
                f'You are a specialized agent designed to answer queries about a document.'
                "You must ALWAYS use at least one of the tools provided when answering "
                "a question; do NOT rely on prior knowledge."
            )
        )

    # ---- Documento como Herramienta ----
    @property
    def document_tool(self) -> QueryEngineTool:
        """Retorna el documento como herramienta."""
        sanitized_title = re.sub(r'[^a-zA-Z0-9]', '_', self.title_of_document.lower())
        return QueryEngineTool(
            query_engine=self.agent_ReAct,
            metadata=ToolMetadata(
                name=f"document_agent_tool",
                description=(
                    f'This tool handles the document "{self.title_of_document}", which is about:\n'
                    f"{self.summary_of_document}\n\n"
                    "Use this tool if you need to answer a question related to this topic."
                ),
            ),
        )



    #############################

    # ---- Cargo los nodos ----
    def _load_nodes(self, pkl_path: str):
        """Carga la lista de nodos desde el archivo .pkl."""
        try:
            with open(pkl_path, 'rb') as file:
                nodes = pickle.load(file)
                if not isinstance(nodes, list) or not all(isinstance(node, BaseNode) for node in nodes):
                    logger.error("El archivo .pkl debe contener una lista de objetos BaseNode o sus herederos.")
                    raise ValueError("El archivo .pkl debe contener una lista de objetos BaseNode o sus herederos.")
                if self.verbose:
                    logger.info(f"Cargados {len(nodes)} nodos desde {pkl_path}")
                # Validar que si los elementos de la lista cargada son de tipo Document, se transformen a TextNode
                documents = [node for node in nodes if isinstance(node, Document)]
                other_nodes = [node for node in nodes if not isinstance(node, Document)]
                new_nodes = Settings.text_splitter(documents)
                all_nodes = new_nodes + other_nodes
                if self.verbose and len(documents) > 0:
                    logger.info(f"Transformados {len(documents)} nodos Document a TextNode")
                return all_nodes
        except FileNotFoundError:
            logger.error(f"No se encontró el archivo {pkl_path}")
            raise FileNotFoundError(f"No se encontró el archivo {pkl_path}")
        except pickle.PickleError as e:
            logger.error(f"Error al cargar el archivo .pkl: {e}")
            raise RuntimeError(f"Error al cargar el archivo .pkl: {e}")