from dataclasses import dataclass


@dataclass
class RagConfiguration:
    retrieval_label: str = "default"
    llm_label: str = "default"

    def get_label(self):
        return self.retrieval_label + "--" + self.llm_label
