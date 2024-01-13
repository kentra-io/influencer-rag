import csv
import os

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import context_precision, faithfulness, answer_relevancy

import config
from common import file_utils
from common.file_utils import createFolderIfNotExists
from evaluations.model.evaluation import Evaluation
from evaluations.evaluations_config import evaluations_config
from model.rag_response import RagResponse


def evaluate_question(query, context, answer):
    context_length = len(context) if context else 0
    if context_length == 0:
        return Evaluation(
            faithfulness=0,
            answer_relevancy=0,
            context_precision=0,
        )

    print(f"Evaluating. \n Query: \n {query} \n Context length: \n {context_length} \n Answer: {answer} ")

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

    return Evaluation(
        faithfulness=evaluation["faithfulness"],
        answer_relevancy=evaluation["answer_relevancy"],
        context_precision=evaluation["context_precision"],
    )


# add entry to questions.csv if there is no entry with given query
def persist_question(channel_handle, query):
    questions_file_path = f"{config.evaluations_dir_path}/{channel_handle}/questions.csv"
    if not is_query_present(questions_file_path, query):
        with open(questions_file_path, "a") as questions_file:
            writer = csv.DictWriter(questions_file, fieldnames=["query", "ground_truth"], quoting=csv.QUOTE_NONNUMERIC,
                                    restval="")
            writer.writerow({"query": query})


def is_query_present(questions_file_path, query):
    if not os.path.exists(questions_file_path):
        createFolderIfNotExists(questions_file_path)
        with open(questions_file_path, "w") as questions_file:
            writer = csv.DictWriter(questions_file, fieldnames=["query", "ground_truth"], quoting=csv.QUOTE_NONNUMERIC,
                                    restval="")
            writer.writeheader()
        return False
    else:
        with open(questions_file_path) as questions_file:
            reader = csv.DictReader(questions_file)
            query_present = False
            for row in reader:
                if row["query"] == query:
                    query_present = True
        print("is_query_present: " + str(is_query_present))
        return query_present


def persist_single_evaluation_result(channel_handle, rag_response: RagResponse):
    label = evaluations_config.get_label()
    joined_context = join_context(rag_response.relevant_movie_chunks)
    file_path = f"{config.evaluations_dir_path}/{channel_handle}/evaluations_{label}.csv"
    file_utils.createFolderIfNotExists(file_path)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a") as evaluation_results_file:
        object_to_persist = {
            "query": rag_response.query,
            "channel_handle": channel_handle,
            "context": joined_context,
            "answer": rag_response.llm_user_response,
            "faithfulness": rag_response.evaluation.faithfulness,
            "answer_relevancy": rag_response.evaluation.answer_relevancy,
            "context_precision": rag_response.evaluation.context_precision
        }
        writer = csv.DictWriter(evaluation_results_file, fieldnames=object_to_persist.keys(),
                                quoting=csv.QUOTE_NONNUMERIC, restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerow(object_to_persist)


def join_context(context):
    result = ""
    for chunk in context:
        result += chunk[0].page_content + "\n"
    return result


def persist_evaluation(response: RagResponse, k):
    print(f"Persisting evaluation for the following configuration: {evaluations_config.get_label()}")
    channel_handle = evaluations_config.channel_handle
    persist_question(channel_handle, response.query)
    persist_single_evaluation_result(channel_handle, response)
