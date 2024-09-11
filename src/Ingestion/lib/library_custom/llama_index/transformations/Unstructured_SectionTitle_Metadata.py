import logging

from llama_index.core.schema import TransformComponent

from ._LinkedListManager import LinkedListManager

logger = logging.getLogger(__name__)
logger.propagate = True

class Unstructured_SectionTitle_Metadata(TransformComponent):
    def __call__(self, nodes, **kwargs):
        logger.info("Processing nodes... (Unstructured_SectionTitle_Metadata)")
        list_manager = LinkedListManager(nodes)
        list_manager.assign_section_titles()
        return list(list_manager.nodes.values())