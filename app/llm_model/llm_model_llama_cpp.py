from llama_cpp import Llama

from app.llm_model.llm_model import LlmModel
from app.model.llm_model_response import LlmModelResponse


class LlmModelLlamaCpp(LlmModel):
    def __init__(self, model_name, local_models_path):
        super().__init__(model_name, local_models_path)

        self.llm = Llama(
            model_path=self.local_models_path + self.model_name,
            n_ctx=32768,
            # n_ctx=8192 for machines with less RAM
            n_batch=512,
            n_threads=7,
            n_gpu_layers=2,
            verbose=False,
            seed=42
        )

    def get_model_response(self, system: str, user_query: str) -> LlmModelResponse:
        prompt = f"<s>[INST] \n{system}\n [/INST]</s>\n{user_query}\n"
        output = self.llm(prompt, echo=True, stream=False, max_tokens=4096)
        # TODO maybe add the stop token: stop=["</s>"]

        llm_full_response = output['choices'][0]['text']
        llm_user_response = llm_full_response.replace(prompt, '')

        return LlmModelResponse(llm_user_response, llm_full_response)
