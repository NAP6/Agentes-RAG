from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.core.schema import Document
import os
import logging


class BaseDirectoryLoader(ABC):
    """
    Abstract base class defining the interface for loading documents.
    """

    def __init__(self,
                 directory: Optional[str] = None,
                 files: Optional[List[str]] = None
                 ):
        """
        Initializes the BaseDirectoryLoader with the specified directory and files.

        :param directory:
        :param files:
        """
        self._files = self._clean_file_list(directory, files)
        self._logger = logging.getLogger(__name__)
        self._logger.propagate = True

    @abstractmethod
    def _load_documents(self, files: List[str]) -> List[Document]:
        """
        Abstract method to be implemented by subclasses to load documents.
        Returns a list of Document instances.
        """
        pass

    def _clean_file_list(self,
                   directory: Optional[str] = None,
                   files: Optional[List[str]] = None
                   ) -> List[str]:
        """
        Set the files to be loaded.
        :param directory:
        :param files:
        :return:
        """
        directory_files = []
        if directory is not None:
            directory_files = [os.path.join(directory, f) for f in os.listdir(directory)]

        files = files or []
        combined_files = directory_files + files

        unique_files = {}
        for file_path in combined_files:
            base_name = os.path.basename(file_path)
            if base_name not in unique_files:
                unique_files[base_name] = file_path

        return list(unique_files.values())

    def load_from(self,
                  directory: Optional[str] = None,
                  files: Optional[List[str]] = None
                  ) -> List[Document]:
        """
        Load documents from the specified directory and files.
        :param directory:
        :param files:
        :return:
        """
        files = self._clean_file_list(directory, files)
        if not files or len(files) == 0:
            self._logger.warning("No files to load.")
        self._logger.info(f"Loading documents from: {files}")
        return self._load_documents(files)

    def __call__(self) -> List[Document]:
        """
        This retruns a list of LlamaIndex Document instances.
        """
        if not self._files or len(self._files) == 0:
            self._logger.warning("No files to load.")
            return []
        self._logger.info(f"Loading documents from: {self._files}")
        return self._load_documents(self._files)
