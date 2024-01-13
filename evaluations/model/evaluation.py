from dataclasses import dataclass


@dataclass
class Evaluation:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float = None  # todo when ground truths are implemented


@dataclass
class EvaluationsConfig:
    channel_handle: str
    retrieval_label: str
    llm_label: str
    evaluations_enabled: bool

    def get_label(self):
        return self.retrieval_label + "--" + self.llm_label
