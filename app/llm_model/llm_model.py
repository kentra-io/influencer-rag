from abc import ABC, abstractmethod

from model.llm_model_response import LlmModelResponse


class LlmModel(ABC):
    def __init__(self, model_name, local_models_path=None):
        self.local_models_path = local_models_path
        self.model_name = model_name

    @abstractmethod
    def get_model_response(self, system: str, user_query: str) -> LlmModelResponse:
        pass
