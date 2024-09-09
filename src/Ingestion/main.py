from llama_index.core import Settings
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.vertex import Vertex

from lib.library_custom.llama_index import VertexIEmbeddings
from lib.DirectoryLoader import Unstructured_UF_Loader
from lib.Watcher import File2NodesWatcher
from lib.library_custom.llama_index.transformations import Unstructured_Medatata_PostProcessor
from lib.library_custom.llama_index.transformations import Unstructured_SectionTitle_Metadata
from lib.library_custom.llama_index.transformations import Unstructured_Filter
from lib.library_custom.langchain.models import GCP_Model

from config import Config, setup_logger

# Cofiguracion
config = Config()

# Set up the logger
logger = setup_logger(config.log_dir)
logger.info("Starting main.py")

# Create a loader instance
loader = Unstructured_UF_Loader(image_path_output=config.images_dir)

# Create ingestion pipeline instance
embedding = VertexIEmbeddings(credentials_path=config.gcp_credentials_path)
Settings.llm = Vertex(model=config.llm_model)
Settings.embed_model = embedding


gcp_model = GCP_Model(model_name=config.llm_model, credentials_path=config.gcp_credentials_path)
unstuc_meta_post_processor = Unstructured_Medatata_PostProcessor(
    gcp_model=gcp_model,
    meta_folder_path=config.meta_dir
)

unstuc_section_title_as_metadata = Unstructured_SectionTitle_Metadata()

unstuc_filter = Unstructured_Filter(filter_block_type=["UncategorizedText"])

semantic_spliter = SemanticSplitterNodeParser(
    buffer_size=1,
    embed_model=embedding,
    include_metadata=True,
    include_prev_next_rel=True
)

qa_extractor = QuestionsAnsweredExtractor(questions=3)

transformation = [
    unstuc_meta_post_processor,
    unstuc_section_title_as_metadata,
    unstuc_filter,
    # json_to_metadata,
    semantic_spliter,
    # sumary_extractor, # Este debe ir antes o despues del semantic splitter?
    qa_extractor,
    embedding
]

pipeline = IngestionPipeline(transformations=transformation)

# Create a watcher instance
# watcher = File2NodesWatcher(
#     watch_directory=config.source_dir,
#     nodes_output_directory=config.out_dir,
#     file_reader=loader
# )
# watcher.start_watch()


import pickle
import os

with open(os.path.join(config.out_dir, 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.temporal.pk'), 'rb') as file:
    nodes = pickle.load(file)

new_nodes = pipeline.run(nodes=nodes)


with open(os.path.join(config.out_dir, 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.temporal2.pk'), 'wb') as file:
    pickle.dump(new_nodes, file)