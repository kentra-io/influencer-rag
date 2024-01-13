from typing import Optional

from llm_model.llm_model import LlmModel
from llm_model.llm_model_llama_cpp import LlmModelLlamaCpp
from llm_model.llm_model_open_ai import LlmModelOpenAI
from llm_model.llm_model_transformers import LlmModelTransformers


def get_llm_model(model_name: str, local_models_path: str) -> Optional[LlmModel]:
    if "gguf" in model_name:
        return LlmModelLlamaCpp(model_name, local_models_path)
    elif "gpt" in model_name:
        return LlmModelOpenAI(model_name)
    else:
        return LlmModelTransformers(model_name)
