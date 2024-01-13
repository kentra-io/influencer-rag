from config import channels
from evaluations.model.evaluation import EvaluationsConfig

evaluations_config = EvaluationsConfig(
    channel_handle=channels[0].handle,
    retrieval_label="default",
    llm_label="default",
    evaluations_enabled = False
)

