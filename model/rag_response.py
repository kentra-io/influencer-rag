from dataclasses import dataclass
from datetime import timedelta
from typing import List

from evaluations.model.evaluation import Evaluation


@dataclass
class RagResponse:
    query: str
    llm_user_response: str
    llm_full_response: str
    relevant_movie_chunks: List
    evaluation: Evaluation
    response_time: timedelta
