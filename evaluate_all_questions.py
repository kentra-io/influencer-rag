import csv

import config
from _3_run_llm_llama_cpp import ask_question
from common.file_utils import createFolderIfNotExists
from evaluations.evaluations import persist_evaluation


def evaluate_all_questions(channel_handle: str):
    createFolderIfNotExists(f"{config.evaluations_dir_path}/{channel_handle}")

    with open(f"{config.evaluations_dir_path}/{channel_handle}/questions.csv") as questions_file:
        questions = csv.DictReader(questions_file)
        for question in questions:
            print(question["query"])
            rag_response = ask_question(question["query"], enable_vector_search=True, k=config.k)
            persist_evaluation(rag_response, k = config.k)


evaluate_all_questions("BenFelixCSI") # todo this channel is hardcoded now, implement other channels
