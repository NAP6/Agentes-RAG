from dataclasses import dataclass

try:
    from lib.Config.Config import _Config
except ModuleNotFoundError:
    from ..lib.Config.Config import _Config

@dataclass
class ConfigApp(_Config):

    def __init__(self):
        super().__init__()

Config = ConfigApp()