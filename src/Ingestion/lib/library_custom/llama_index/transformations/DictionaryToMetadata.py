import json
import os
from typing import Any
import logging

from llama_index.core.schema import TransformComponent
from llama_index.core.bridge.pydantic import Field

logger = logging.getLogger(__name__)
logger.propagate = True

class DictionaryToMetadata(TransformComponent):

    meta_folder_path: str = Field(default=None, description="Path to the folder containing metadata files")

    def __init__(self,
                 meta_folder_path: str,
                 **kwargs: Any
                 ):
        super().__init__(**kwargs)
        self.meta_folder_path = meta_folder_path
        logger.info("DictionaryToMetadata initialized")

    def __call__(self, nodes, **kwargs):
        logger.info("Processing nodes... (DictionaryToMetadata)")
        for node in nodes:
            meta_path = os.path.join(self.meta_folder_path, f"{node.metadata['title_of_the_document']}.json")
            dictionary = self._load_metadata(meta_path)
            node.metadata.update(dictionary)
        return nodes

    def _load_metadata(self, json_path: str) -> dict:
        """Load metadata from a folder."""
        try:
            with open(json_path, 'r') as archivo:
                metadata = archivo.read()
                logger.debug("JSON metadata loaded successfully")
        except Exception as e:
            logger.error("Failed to load JSON metadata from %s: %s", json_path, e)
            metadata = ""
        return json.loads(metadata)
