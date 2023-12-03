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

def ask_question(question):
    query_start_time = time.time()

    similar_documents = chroma.similarity_search(question, top_k=3)
    similar_documents_combined = " ".join([doc.page_content for doc in similar_documents])

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
        return [answer.group(1), timedelta(seconds=execution_time), response]
    else:
        return ['', timedelta(seconds=execution_time), response]

def print_response(response):
    if response[0] != '':
        print(f"Chatbot: '{response[0]}'; generated in {response[1]}\n")
    else:
        print("Error! No 'Answer' present in chatbot's response: ", response[2])


if len(sys.argv) > 1:
    response = ask_question(sys.argv[1])
    print_response(response)
else:
    while True:
        print("Welcome to the chat, type 'exit' to leave.\n")

        user_query = input("You: ")

        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response = ask_question(user_query)
        print_response(response)
