import logging
import traceback
import os
from typing import Sequence, Any
import concurrent.futures
import time

from llama_index.core.schema import TransformComponent, BaseNode, NodeRelationship
from llama_index.core.bridge.pydantic import Field

from ...langchain.models import GCP_Model
from ...langchain.prompts.text_block_evaluate import create_chain_evaluation
from ...langchain.prompts.img_to_text import transcriber_job_description_prompt, image_summary_transcriber_prompt

logger = logging.getLogger(__name__)
logger.propagate = True

class Unstructured_Medatata_PostProcessor(TransformComponent):

    gcp_model: GCP_Model = None
    chain_evaluator: Any = Field(default=None, description="Chain evaluator for evaluating text blocks")
    meta_folder_path: str = Field(default=None, description="Path to the folder containing metadata files")

    def __init__(self,
                  gcp_model: GCP_Model,
                  meta_folder_path: str,
                  **kwargs: Any
                  ):
        super().__init__(**kwargs)
        self.gcp_model = gcp_model
        self.meta_folder_path = meta_folder_path
        self.chain_evaluator = create_chain_evaluation(self.gcp_model)
        logger.info("Unstructured_Medatata_PostProcessor initialized")

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        logger.info("Processing nodes... (Unstructured_Medatata_PostProcessor)")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._reclasificar_bloque_con_reintentos, node, nodes) for node in nodes]
            new_nodes = [future.result() for future in concurrent.futures.as_completed(futures)]
        return new_nodes

    def _reclasificar_bloque_con_reintentos(self, node: BaseNode, nodes: Sequence[BaseNode], intentos=0):
        try:
            return self._reclasificar_bloque(node, nodes)
        except Exception as e:
            intentos += 1
            if intentos < 20:
                logger.error(f"Error al reclasificar el nodo {node.node_id}: {e}. Reintentando en 10 segundos (Intento {intentos}/20)...")
                time.sleep(10)
                return self._reclasificar_bloque_con_reintentos(node, nodes, intentos)
            else:
                logger.error(f"Intento final fallido para el nodo {node.node_id} después de 20 reintentos: {e}. No se procesará más este nodo.")
                return node  # Devuelve el nodo sin modificar después de los 20 intentos fallidos.

    def _reclasificar_bloque(self, node: BaseNode, nodes: Sequence[BaseNode]) -> BaseNode:
        if not node.metadata.get('block_type'):
            logger.error(f"Node '{node.node_id}' does not have 'block_type' metadata key.")
            return node

        node_block_type = node.metadata['block_type']
        node_next_text = self._get_next_node_text(node, nodes)

        try:
            if node_block_type == 'Table':
                return self._process_table_node(node)
            elif node_block_type == 'Image':
                return self._process_image_node(node)
            else:
                return self._evaluate_block_type(node, node_next_text)
        except Exception as e:
            error_message = ''.join(traceback.format_exception(None, e, e.__traceback__))
            logger.error(f"An error occurred during classification: {e}\n{error_message}")

        return node

    def _get_next_node_text(self, node, nodes):
        """Get the text of the next node based on relationships."""
        next_node_relation = node.relationships.get(NodeRelationship.NEXT)
        if not next_node_relation:
            return None
        next_node_id = next_node_relation.node_id
        next_node = next(filter(lambda x: x.node_id == next_node_id, nodes), None)
        return next_node.text if next_node else None

    def _process_table_node(self, node):
        """Process a node identified as a 'Table'."""
        if not node.metadata.get('image_path'):
            logger.error(f"Node '{node.node_id}' does not have 'image_path' metadata key.")
            return node

        node.text = self.gcp_model.invoke(transcriber_job_description_prompt, image_path=node.metadata['image_path'])
        node.metadata.update({'makes_sense': True, 'description': None})
        return node

    def _process_image_node(self, node):
        """Process a node identified as an 'Image'."""
        if not node.metadata.get('image_path'):
            logger.error(f"Node '{node.node_id}' does not have 'image_path' metadata key.")
            return node

        node.text = self.gcp_model.invoke(image_summary_transcriber_prompt, image_path=node.metadata['image_path'])
        node.metadata.update({'makes_sense': True, 'description': None})
        return node

    def _evaluate_block_type(self, node, node_next_text):
        """Evaluate the block type for nodes that are neither 'Table' nor 'Image'."""
        meta_path = os.path.join(self.meta_folder_path, f"{node.metadata['title_of_the_document']}.json")
        input_row = {
            "text_to_evaluate": node.text,
            "old_type": node.metadata['block_type'],
            'next_block': node_next_text,
            'meta': self._load_metadata(meta_path)
        }
        block_type_model = self.chain_evaluator.invoke(input_row)
        node.metadata.update(block_type_model)
        return node

    def _load_metadata(self, json_path: str) -> str:
        """Load metadata from a folder."""
        try:
            with open(json_path, 'r') as archivo:
                metadata = archivo.read()
                logger.debug("JSON metadata loaded successfully")
        except Exception as e:
            logger.error("Failed to load JSON metadata from %s: %s", json_path, e)
            metadata = ""
        return metadata

