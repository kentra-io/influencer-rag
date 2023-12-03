# This 'readline' import below must be here, so the input(...) function prints it's output to stdout, not stderr
#   This looks like some weird Python bug
import readline
import re
import sys
import time
from datetime import timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers

from vector_db import chroma_provider

chroma = chroma_provider.get_chroma()

# Load model and tokenizer
model_name = "ericzzz/falcon-rw-1b-instruct-openorca"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def ask_question(question, enable_vector_search):
    query_start_time = time.time()

    if enable_vector_search:
        similar_documents = chroma.similarity_search(question, top_k=3)
        similar_documents_combined = " ".join([doc.page_content for doc in similar_documents])
    else:
        similar_documents = None
        similar_documents_combined = ""

    pipeline = transformers.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    prompt = ("<SYS>You'll get a question and context. Base your answer on the context provided."
              f"<INST>Context: {similar_documents_combined}\n"
              f"Question: {question}"
              "<RESP>")

    response = pipeline(
        prompt,
        max_length=1000,
        repetition_penalty=1.05,
        pad_token_id=50256
    )

    execution_time = time.time() - query_start_time

    answer = re.search(r'.*<RESP>(.*)', response[0]['generated_text'])

    if answer:
        return [answer.group(1), timedelta(seconds=execution_time), response, similar_documents]
    else:
        return ['', timedelta(seconds=execution_time), response, similar_documents]


def print_response(response):
    if response[0] != '':
        print(f"Chatbot: '{response[0]}'; generated in {response[1]}\n")
    else:
        print("Error! No 'Answer' present in chatbot's response: ", response[2])


def main():
    if len(sys.argv) > 1:
        response = ask_question(sys.argv[1], True)
        print_response(response)
    else:
        print("Welcome to the chat. Type your query or one of the following commands:\n"
              "- 'response' to see the entire response of the previous query,\n"
              "- 'context' to see the context,\n"
              "- 'enable' to enable vector db search,\n"
              "- 'disable' to disable vector db search,\n"
              "- 'exit' to leave.\n")
        response = []
        vector_db_search_enabled = True
        while True:
            user_query = input("You: ")

            if user_query.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            elif user_query.lower() == 'response':
                print(response[2], "\n")
            elif user_query.lower() == 'context':
                for result in response[3]:
                    print(f"Content: '{result.page_content}'")
                    print(f"- Metadata: '{result.metadata}'")
                print("\n")
            elif user_query.lower() == 'enable':
                vector_db_search_enabled = True
            elif user_query.lower() == 'disable':
                vector_db_search_enabled = False
            else:
                response = ask_question(user_query, vector_db_search_enabled)
                print_response(response)


if __name__ == "__main__":
    main()
