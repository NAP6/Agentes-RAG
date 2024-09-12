import logging
import os
import pickle

from llama_index.core import Settings
from llama_index.core.extractors import QuestionsAnsweredExtractor, KeywordExtractor, SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser

from lib.DirectoryLoader import Unstructured_UF_Loader
from lib.Watcher import File2NodesWatcher
from lib.library_custom.llama_index.transformations import Unstructured_Medatata_PostProcessor
from lib.library_custom.llama_index.transformations import Unstructured_SectionTitle_Metadata
from lib.library_custom.llama_index.transformations import Unstructured_Filter
from lib.library_custom.llama_index.transformations import DictionaryToMetadata
from lib.library_custom.langchain.models import GCP_Model

from config import Config

logger = logging.getLogger()
logger.propagate = True

logger.info("Starting Ingestion")

# Create a loader instance
loader = Unstructured_UF_Loader(image_path_output=Config.images_dir)
logger.info("Loader created")


# Create a GCP model instance this is an LLM for langchain
gcp_model = GCP_Model(model_name=Config.llm.model)
logger.info("GCP Model created")

# Define the transformations
transformation = [

    Unstructured_Medatata_PostProcessor(
        gcp_model=gcp_model,
        meta_folder_path=Config.meta_dir
    ),

    Unstructured_SectionTitle_Metadata(),

    Unstructured_Filter(filter_block_type=["UncategorizedText"]),

    DictionaryToMetadata(meta_folder_path=Config.meta_dir),

    SemanticSplitterNodeParser( # Tenemos que separar las tablas
        buffer_size=1,
        embed_model=Config.embed_model,
        include_metadata=True,
        include_prev_next_rel=True
    ),

    # SummaryExtractor(llm=Config.llm), # No funciona

    # KeywordExtractor(), # No funciona

    # QuestionsAnsweredExtractor(questions=3), # No funciona

    Config.embed_model
]

# Create a pipeline instance
pipeline = IngestionPipeline(transformations=transformation)

# Create a watcher instance
watcher = File2NodesWatcher(
    watch_directory=Config.source_dir,
    nodes_output_directory=Config.out_dir,
    file_reader=loader,
    ingestion_pipeline=pipeline
)
# watcher.start_watch()

# file_name = 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf'
# file_path = os.path.join(Config.source_dir, file_name)
#
# logger.info(f'Procesando archivo: {file_path}')
# nodes = loader.load_from(files=[file_path])
# logger.info('Ejecutando el pipeline de ingestión')
# nodes = pipeline.run(documents=nodes)
#
# output_file_path = os.path.join(Config.out_dir, f'{file_name}.pk')
# with open(output_file_path, 'wb') as file:
#     pickle.dump(nodes, file)
# logger.info(f'Nodos guardados en: {output_file_path}')
