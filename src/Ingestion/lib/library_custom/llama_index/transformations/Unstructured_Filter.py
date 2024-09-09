from typing import List, Any

from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field

from ._LinkedListManager import LinkedListManager


class Unstructured_Filter(TransformComponent):
    filter_block_type: List[str] = Field(default=None, description="Path to the folder containing metadata files")

    def __init__(self,
                  filter_block_type: List[str],
                  **kwargs: Any
                  ):
        super().__init__(**kwargs)
        self.filter_block_type = filter_block_type

    def __call__(self, nodes, **kwargs):
        list_manager = LinkedListManager(nodes)
        list_manager.remove_nodes_by_block_type(self.filter_block_type)
        return list(list_manager.nodes.values())