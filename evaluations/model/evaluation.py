from dataclasses import dataclass

from evaluations.model.rag_configuration import RagConfiguration


@dataclass
class Evaluation:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    rag_configuration: RagConfiguration
    context_recall: float = None # todo when ground truths are implemented
