from itertools import count
from typing import Sequence

from llama_index.core.schema import NodeRelationship, BaseNode


class LinkedListManager:
    def __init__(self, nodes: Sequence[BaseNode]):
        self.nodes = {node.node_id: node for node in nodes}  # Crear un diccionario de nodos por ID

    def get_node_by_id(self, node_id: str) -> BaseNode:
        return self.nodes.get(node_id)

    def find_nearest_title_parent(self, node: BaseNode) -> BaseNode:
        current_node = node.relationships.get(NodeRelationship.PREVIOUS)
        if current_node is None:
            return None
        current_node_id = current_node.node_id

        while current_node_id is not None:
            current_node = self.get_node_by_id(current_node_id)
            if current_node is None:
                return None
            if current_node.metadata['block_type'] == 'Title':
                return current_node
            current_node = current_node.relationships.get(NodeRelationship.PREVIOUS)
            if current_node is None:
                return None
            current_node_id = current_node.node_id

        return None

    def assign_section_titles(self):
        for node_id, node in self.nodes.items():
            nearest_title_node = self.find_nearest_title_parent(node)
            if nearest_title_node:
                node.metadata['section_title'] = nearest_title_node.text

    def remove_nodes_by_block_type(self, block_types):
        nodes_to_remove = {node_id: node for node_id, node in self.nodes.items() if node.metadata['block_type'] in block_types}
        for node_id, node in nodes_to_remove.items():
            prev_relation = node.relationships.get(NodeRelationship.PREVIOUS)
            next_relation = node.relationships.get(NodeRelationship.NEXT)

            if prev_relation and prev_relation.node_id in self.nodes:
                self.nodes[prev_relation.node_id].relationships[NodeRelationship.NEXT] = next_relation

            if next_relation and next_relation.node_id in self.nodes:
                self.nodes[next_relation.node_id].relationships[NodeRelationship.PREVIOUS] = prev_relation

            # Eliminar el nodo actual del diccionario
            del self.nodes[node_id]