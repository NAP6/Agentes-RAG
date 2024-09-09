from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Image
)

class Image2Text_Model:

    def __init__(self, model_name):
        self.model_name = model_name
        self.__model = GenerativeModel(model_name)