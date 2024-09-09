from llama_index.core.schema import TransformComponent

from ._LinkedListManager import LinkedListManager


class Unstructured_SectionTitle_Metadata(TransformComponent):
    def __call__(self, nodes, **kwargs):
        list_manager = LinkedListManager(nodes)
        list_manager.assign_section_titles()
        return list(list_manager.nodes.values())