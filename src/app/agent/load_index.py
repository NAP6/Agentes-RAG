import pickle
from typing import Optional, Sequence

from attr.validators import instance_of
from fontTools.misc.cython import returns
from llama_index.core import SummaryIndex, VectorStoreIndex, Settings
from llama_index.core.data_structs import IndexList
from llama_index.core.schema import BaseNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llms import LLM


class NodeLoaderAgent:

    def __init__(self,
                 pkl_path: str,
                 llm: Optional[LLM] = Settings.llm,
                 verbose: Optional[bool] = False
                 ):
        self.llm = llm
        self.verbose = verbose
        nodes: Sequence[BaseNode] = self._load_nodes(pkl_path)
        self.title_of_the_document = nodes[0].metadata.get('title_of_the_document')
        self.vector_qe = VectorStoreIndex(nodes).as_query_engine(llm=self.llm)
        self.summary_qe = SummaryIndex(nodes).as_query_engine(llm=self.llm)
        self.summary = self._create_document_summary()
        self.query_engine_tools = self._create_tools()
        # self.agent = self._create_agent()

    def _load_nodes(self, pkl_path: str):
        """Carga la lista de nodos desde el archivo .pkl."""
        try:
            with open(pkl_path, 'rb') as file:
                nodes = pickle.load(file)
                if not isinstance(nodes, list) or not all(isinstance(node, BaseNode) for node in nodes):
                    raise ValueError("El archivo .pkl debe contener una lista de objetos BaseNode o sus herederos.")
                if self.verbose:
                    print(f"Cargados {len(nodes)} nodos desde {pkl_path}")
                return nodes
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo {pkl_path}")
        except pickle.PickleError as e:
            raise RuntimeError(f"Error al cargar el archivo .pkl: {e}")

    def _create_tools(self):
        """Crea las herramientas de consulta para el agente."""
        query_engine_tools = [
            QueryEngineTool(
                query_engine=self.vector_qe,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" '{self.title_of_the_document}' (e.g. specific sections, key arguments,"
                        " research findings, literature reviews, or more)"
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=self.summary_qe,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f" of EVERYTHING about '{self.title_of_the_document}'. For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            ),
        ]

        if self.verbose:
            print(f"Herramientas de consulta creadas: {len(query_engine_tools)}")

        return query_engine_tools

    def _create_document_summary(self) -> str:
        """Crea un resumen general del documento."""
        return self.summary_qe.query("Give me a summary").response

    def _create_agent(self):
        """Crea un agente configurado con las herramientas de consulta y el LLM."""
        # agent = OpenAIAgent.from_tools(
        #     self.query_engine_tools,
        #     llm=self.llm,
        #     verbose=self.verbose,
        #     system_prompt=self._description_like((
        #         "You are a specialized agent designed to answer queries about <TOPIC>."
        #         "You must ALWAYS use at least one of the tools provided when answering "
        #         "a question; do NOT rely on prior knowledge."
        #     ))
        # )

        if self.verbose:
            print("Agente creado con éxito.")

        # return agent
        pass

    def run(self):
        """Método para iniciar el agente y realizar consultas."""
        if self.verbose:
            print("Ejecutando el agente...")
        # Aquí puedes definir cómo ejecutar consultas con el agente, dependiendo de tu implementación.
