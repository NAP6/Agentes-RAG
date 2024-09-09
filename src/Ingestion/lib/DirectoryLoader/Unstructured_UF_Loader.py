import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import Document
import pandas as pd
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from pandas import DataFrame

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_dataframe

from .BaseDirectoryLoader import BaseDirectoryLoader


class Unstructured_UF_Loader(BaseDirectoryLoader):
    """
    Loader class that inherits from UniqueFileInSourceLoader to implement the loading
    of unstructured files from a source directory that do not exist in a target directory.

    Attributes:
        source_dir (str): Directory to check for unique files.
        target_dir (str): Directory against which the source directory files are compared.
        unique_files_in_source (List[str]): List of paths to unique files found in source_dir.
    """

    def __init__(self,
                 directory: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 image_path_output: str = None
                 ):
        """
        Initializes the BaseDirectoryLoader with the specified directory and files.

        :param directory:
        :param files:
        :param image_path_output:
        """
        try:
            os.makedirs(image_path_output, exist_ok=True)
            self._image_path_output = image_path_output
        except Exception as e:
            # Manejar cualquier excepción que pueda surgir al intentar crear el directorio
            raise OSError(f"No se pudo crear el directorio {image_path_output}: {e}")

        super().__init__(directory, files)

    def _df_to_documents(self, df: DataFrame) -> List[Document]:
        """
        Converts a DataFrame to a list of Document instances.

        :param df:
        :return:
        """
        required_columns = ['text', 'type', 'filetype', 'languages', 'page_number', 'filename', 'title', 'image_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltan las siguientes columnas requeridas en el DataFrame: {missing_columns}")

        df['node_id'] = df.apply(lambda row: f"{row.name}_{row['title']}", axis=1)
        df['previous_node_id'] = df['node_id'].shift(1)
        df['next_node_id'] = df['node_id'].shift(-1)

        def create_document(row):
            document = Document(
                id_=row['node_id'],
                text=row['text'],
                metadata={
                    'block_type': row['type'],
                    'file_type': row['filetype'],
                    'languages': row['languages'],
                    'page_number': row['page_number'],
                    'file_name': row['filename'],
                    'title_of_the_document': row['title'],
                    'image_path': row['image_path']
                },
                excluded_llm_metadata_keys=[],
                excluded_embed_metadata_keys=[]
            )
            if pd.notna(row['previous_node_id']):
                document.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=row['previous_node_id']
                )
            if pd.notna(row['next_node_id']):
                document.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=row['next_node_id']
                )
            return document

        df['documents'] = df.apply(create_document, axis=1)
        return df['documents'].tolist()

    def _read_file(self, file_path: str) -> List[Document]:
        """
        Reads the file at the specified path and returns a list of Document instances.

        :param file_path:
        :return:
        """
        base_name = os.path.basename(file_path)
        file_name, file_ext = os.path.splitext(base_name)
        if file_ext.lower() != '.pdf':
            return []

        elements = partition_pdf(
            filename=file_path,  # mandatory
            strategy="hi_res",  # mandatory to use ``hi_res`` strategy
            extract_images_in_pdf=True,  # mandatory to set as ``True``
            extract_image_block_types=["Image", "Table"],  # optional
            extract_image_block_to_payload=False,  # optional
            extract_image_block_output_dir=os.path.join(self._image_path_output, file_name),
            # optional - only works when ``extract_image_block_to_payload=False``
        )
        df: DataFrame = convert_to_dataframe(elements)
        df['title'] = file_name

        return self._df_to_documents(df)

    def _load_documents(self, files: List[str]) -> List[Document]:
        """
        Loads unstructured documents from the unique files in the source directory
        using multithreading to improve performance.

        Returns:
            List[Document]: A list of Document instances.
        """
        documents = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(self._read_file, files)
            for result in results:
                documents.extend(result)

        return documents
