import csv
import os

from app import config

from app._3_run_llm_llama_cpp import ask_question
from app.common.file_utils import createFolderIfNotExists
from app.evaluations.evaluations import persist_evaluation
from app.evaluations.evaluations_config import evaluations_config


def main():
    print(
        f"Starting evaluations for channel: {evaluations_config.channel_handle}). Configuration: {evaluations_config}")
    questions_file_path = f"{config.evaluations_dir_path}/{evaluations_config.channel_handle}/questions.csv"
    createFolderIfNotExists(questions_file_path)
    if not os.path.exists(questions_file_path):
        print("In order to evaluate all questions, you need to provide a questions.csv file in the evaluations "
              "directory. This will be done automatically when you start asking questions while you have evaluations "
              "enabled. To enable it, set evaluations_enabled = True in config. Check evaluations/evaluations.md for more.")
        print("You can also create the file manually and add questions to it.")
        print(f"evaluations directory path: {config.evaluations_dir_path}")
        exit(1)

    with open(questions_file_path) as questions_file:
        questions = csv.DictReader(questions_file)
        for question in questions:
            print(question["query"])
            rag_response = ask_question(question["query"], enable_vector_search=True, k=config.k)
            persist_evaluation(rag_response, k=config.k)


if __name__ == "__main__":
    main()
