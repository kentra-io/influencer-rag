# This 'readline' import below must be here, so the input(...) function prints it's output to stdout, not stderr
#   This looks like some weird Python bug
import readline
import re
import time
from datetime import timedelta

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Chroma client
chroma = Chroma(persist_directory="../chroma_db", collection_name="youtube_transcripts",
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                )

# Load model and tokenizer
model_name = "ericzzz/falcon-rw-1b-instruct-openorca"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def ask_question(question):
    # Perform retrieval from Chroma
    similar_documents = chroma.similarity_search(question, top_k=3)
    # print("Retrieved context: " + " \n".join(
    #     [doc.page_content for doc in similar_documents]))

    # Combine retrieved context with the user query
    combined_input = "[INST]You'll get a question and context. Base your answer only on the context provided. \n" \
                     " Context: " + " ".join(
    [doc.page_content for doc in similar_documents]) + "\n Question: " + question + "[/INST]"

    # Encode the input and generate a response
    inputs = tokenizer.encode(combined_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

print("Welcome to the chat, type 'exit' to leave.\n")

while True:
    user_query = input("You: ")

    if user_query.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    query_start_time = time.time()
    response = ask_question(user_query)
    execution_time = time.time() - query_start_time

    answer = re.search(r'Answer: (.*)', response)

    if answer:
        print(f"Chatbot: '{answer.group(1)}'; generated in {timedelta(seconds=execution_time)}\n")
    else:
        print("Error! No answer from chatbot")
