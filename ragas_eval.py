from ragas import evaluate
from datasets import Dataset
from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall


def evaluate_question(query, context, answer):
    print(f"Evaluating. \n Query: \n {query} \n Context: \n {context} \n Answer: {answer} ")

    dataset = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [context],
        },
    )

    evaluation = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            # context_recall
        ]
    )

    return evaluation


class Evaluation:
    def __init__(self, faithfulness=None, answer_relevancy=None, context_precision=None, context_recall=None):
        self.faithfulness = faithfulness
        self.answer_relevancy = answer_relevancy
        self.context_precision = context_precision
        self.context_recall = context_recall

    def __str__(self):
        return (f"Evaluation("
                f"faithfulness={self.faithfulness}, "
                f"answer_relevancy={self.answer_relevancy}, "
                f"context_precision={self.context_precision}, "
                f"context_recall={self.context_recall}")
