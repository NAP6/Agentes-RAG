import os
from dataclasses import dataclass

from lib.Config.Config import _Config
from sqlalchemy.testing.plugin.plugin_base import logging


@dataclass
class _ConfigIngestion(_Config):

    _source_dir: str = 'raw_files'
    _out_dir: str = 'nodes'
    _images_dir: str = 'img'
    _meta_dir: str = 'metadata'

    def __init__(self):
        self._log_level = logging.INFO
        super().__init__()

    # ---- Source Directory ----
    @property
    def source_dir(self) -> str:
        """Get the source directory."""
        path = os.path.join(self.base_dir, self._source_dir)
        os.makedirs(path, exist_ok=True)
        return path

    # ---- Output Directory ----
    @property
    def out_dir(self) -> str:
        """Get the output directory."""
        path = os.path.join(self.base_dir, self._out_dir)
        os.makedirs(path, exist_ok=True)
        return path

    # ---- Images Directory ----
    @property
    def images_dir(self) -> str:
        """Get the images directory."""
        path = os.path.join(self.base_dir, self._images_dir)
        os.makedirs(path, exist_ok=True)
        return path

    # ---- Metadata Directory ----
    @property
    def meta_dir(self) -> str:
        """Get the metadata directory."""
        path = os.path.join(self.base_dir, self._meta_dir)
        os.makedirs(path, exist_ok=True)
        return path

Config = _ConfigIngestion()