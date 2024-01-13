import re

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_model.llm_model import LlmModel
from model.llm_model_response import LlmModelResponse


class LlmModelTransformers(LlmModel):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_model_response(self, system: str, user_query: str) -> LlmModelResponse:
        pipeline = transformers.pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

        prompt = f"<SYS> {system} <INST> {user_query} <RESP> "

        response = pipeline(
            prompt,
            max_length=1000,
            repetition_penalty=1.05,
            pad_token_id=50256
        )

        answer = re.search(r'.*<RESP> (.*)', response[0]['generated_text'])

        if answer:
            return LlmModelResponse(answer.group(1), response)
        else:
            return LlmModelResponse("", response)
