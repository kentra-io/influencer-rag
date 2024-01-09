from dataclasses import dataclass

from model.channel import Channel
from vector_db.vector_db_model import VectorDbType


@dataclass
class VectorDbConfig:
    max_score: int

channels = [
    Channel('WesRoth', 'UCqcbQf6yw5KzRoDDcZ_wBSw'),
    Channel('PromptEngineer48', 'UCX6c6hTIqcphjMsXbeanJ1g'),
    Channel('engineerprompt', 'UCDq7SjbgRKty5TgGafW8Clg'),
    # Channel('BenFelixCSI', 'UCDXTQ8nWmx_EhZ2v-kp7QxA')
]

default_vector_db = VectorDbType.CHROMA
vector_db_configs = {
    VectorDbType.CHROMA: VectorDbConfig(0.6),
    VectorDbType.ELASTICSEARCH: VectorDbConfig(0.9)
}

transcripts_dir_path = "data/transcripts"
evaluations_dir_path = "data/evaluations"
local_models_path = "../models/"
evaluations_enabled = False

k = 2

# disable parallelism for tokenizers to silence the warnings
TOKENIZERS_PARALLELISM=False

# model_name = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
model_name = "gpt-3.5-turbo-1106"  # Requires OPENAI_API_KEY
# model_name = "ericzzz/falcon-rw-1b-instruct-openorca"
