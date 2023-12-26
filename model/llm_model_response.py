from dataclasses import dataclass


@dataclass
class LlmModelResponse:
    llm_user_response: str
    llm_full_response: str
