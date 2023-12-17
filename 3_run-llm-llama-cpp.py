import json
import sys
import time
import config
from datetime import timedelta

from llama_cpp import Llama

from vector_db import chroma_provider

model = config.model_path
llm = Llama(
    model_path=model,
    n_ctx=32768,
    # n_ctx=8192 for machines with less RAM
    n_batch=512,
    n_threads=7,
    n_gpu_layers=2,
    verbose=False,
    seed=42
)

chroma = chroma_provider.get_chroma()


def read_full_movie_transcript(file_path, movie_title):
    with open(file_path, 'r') as file:
        movies = json.load(file)

    for movie in movies:
        if movie['title'] == movie_title:
            return movie['transcript']

    return None


def prepare_transcription_fragments(relevant_movie_chunks):
    movie_fragments_for_llm = []
    full_movies_already_added = []

    for movie_chunk in relevant_movie_chunks:
        document, score = movie_chunk

        if score < 0.5:
            movie_metadata = document.metadata

            if movie_metadata['istitle'] and (movie_metadata['title'] not in full_movies_already_added):
                movie_fragments_for_llm.append(
                    f"Movie title: '{movie_metadata['title']}', "
                    f"movie transcript: "
                    f"'{read_full_movie_transcript(movie_metadata['file'], movie_metadata['title'])}'; ")
                full_movies_already_added.append(movie_metadata['title'])
            elif not movie_metadata['istitle']:
                movie_fragments_for_llm.append(
                    f"Movie title: '{movie_metadata['title']}', "
                    f"movie transcript: '{document.page_content}'; ")

    if len(movie_fragments_for_llm) > 0:
        return "\n".join(movie_fragments_for_llm)
    else:
        return None


def ask_question(question, enable_vector_search):
    query_start_time = time.time()

    if enable_vector_search:
        relevant_movie_chunks = chroma.similarity_search_with_score(question, 10)
        relevant_movies = prepare_transcription_fragments(relevant_movie_chunks)
    else:
        relevant_movie_chunks = None
        relevant_movies = None

    if relevant_movies is not None:
        system = ("You'll get a question and a series of Youtube movie transcripts. "
                  "Each movie will contain the title and the relevant transcription fragment. "
                  "Base your answer only on the data coming from provided transcriptions and "
                  "make sure to always include the movie title in your answer."
                  f" List of movies: {relevant_movies}")
    else:
        system = ("First, say explicitly that you haven't found any relevant information "
                  "in the movie library. Then answer the following question "
                  "based on your knowledge.")

    prompt = f"<s>[INST] \n{system}\n [/INST]</s>\n{question}\n"

    output = llm(prompt, echo=True, stream=False, max_tokens=4096)
    # TODO maybe add the stop token: stop=["</s>"]
    # print(f"output: {output}")

    execution_time = time.time() - query_start_time

    response = output['choices'][0]['text']

    answer = response.replace(prompt, '')

    if answer:
        return [answer, timedelta(seconds=execution_time), response, relevant_movie_chunks]
    else:
        return ['', timedelta(seconds=execution_time), response, relevant_movie_chunks]


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
            user_query = input("You: ").strip()

            if user_query.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            elif user_query.lower() == 'response':
                print(response[2], "\n")
            elif user_query.lower() == 'context':
                for chunk in response[3]:
                    document, score = chunk
                    print(f"- Content: '{document.page_content}' ({score})")
                    print(f"- Metadata: '{document.metadata}'\n")
            elif user_query.lower() == 'enable':
                vector_db_search_enabled = True
            elif user_query.lower() == 'disable':
                vector_db_search_enabled = False
            else:
                response = ask_question(user_query, vector_db_search_enabled)
                print_response(response)


if __name__ == "__main__":
    main()
