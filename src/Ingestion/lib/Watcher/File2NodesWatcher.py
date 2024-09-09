from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List
import pickle
import os

from .BaseWatcher import BaseWatcher
from ..DirectoryLoader.BaseDirectoryLoader import BaseDirectoryLoader
from llama_index.core.ingestion import IngestionPipeline


class File2NodesWatcher(BaseWatcher):

    def __init__(self,
                 watch_directory: str,
                 nodes_output_directory: str,
                 file_reader: BaseDirectoryLoader,
                 ingestion_pipeline: Optional[IngestionPipeline] = None,
                 **kwargs):

        super().__init__(watch_directory, **kwargs)
        self._nodes_output_directory = nodes_output_directory
        self._file_reader = file_reader
        self._ingestion_pipeline = ingestion_pipeline

        unique_files = self._get_list_of_unique_files_in_source()
        self._process_multiple_files(unique_files)



    def on_created(self, event):
        print(f'1 Archivo creado: {event.src_path}')
        if not event.is_directory:
            self._logger.info(f'Archivo creado: {event.src_path}')
            print(f'Archivo creado: {event.src_path}')
            self._process_and_save_node(event.src_path)

    def _process_and_save_node(self, file_path: str):
        """
        Processes a single file to generate nodes, possibly processes them through a pipeline,
        and saves them into a serialized format in the designated output directory.
        """
        self._logger.info(f'Procesando archivo: {file_path}')
        file_name = Path(file_path).stem
        nodes = self._file_reader.load_from(files=[file_path])
        if self._ingestion_pipeline:
            self._logger.info('Ejecutando el pipeline de ingestión')
            nodes = self._ingestion_pipeline.run(documents=nodes)

        output_file_path = os.path.join(self._nodes_output_directory, f'{file_name}.pk')
        with open(output_file_path, 'wb') as file:
            pickle.dump(nodes, file)
        self._logger.info(f'Nodos guardados en: {output_file_path}')

    def _process_multiple_files(self, file_paths: List[str]):
        """
        Processes multiple files to generate nodes by utilizing a ThreadPoolExecutor
        to parallelize the operation across several threads.
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self._process_and_save_node, file_paths)

    def _get_list_of_unique_files_in_source(self) -> List[str]:
        """
        Identifies and returns a list of files that are present in the source directory
        but not in the target directory. This is achieved by comparing file names
        without considering file extensions.

        Returns:
            List[str]: A list of file paths to unique files in the source directory.
        """
        # Dictionary to hold file names without extensions from source directory
        source_files = {Path(f).stem: f for f in os.listdir(self._watch_directory)}

        # Set of file names without extensions from target directory
        target_files = {Path(f).stem for f in os.listdir(self._nodes_output_directory)}

        # Filter out files that do not have a corresponding file in the target directory
        unique_files = [os.path.join(self._watch_directory, source_files[file_stem])
                        for file_stem in source_files if file_stem not in target_files]

        self._logger.info(f'Archivos únicos en el directorio fuente: {len(unique_files)}')
        return unique_files