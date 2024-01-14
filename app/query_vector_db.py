import sys
import time
from datetime import timedelta

from app.vector_db import chroma_provider


def query_chroma(chroma, prompt):
    print(f"Question: '{prompt}'\n")

    start_time = time.time()
    results = chroma.similarity_search_with_score(prompt, 10)
    execution_time = time.time() - start_time

    print(f"Generated the following answers in {timedelta(seconds=execution_time)}:\n")

    for result in results:
        document, score = result
        if(score < 0.5):
            print(f"Score: '{score}'")
            print(f"Content: '{document.page_content}'")
            print(f"Metadata: '{document.metadata}'\n")
        else:
            print(f"Result irrelevant, score {score}")

    print("\n")


def main():
    chroma = chroma_provider.get_chroma()

    if len(sys.argv) > 1:
        query_chroma(chroma, sys.argv[1])
    else:
        print("Entering chatbot mode. Type 'exit' to quit.")
        while True:
            user_input = input("Enter your prompt: ").strip()
            if user_input.lower() == 'exit':
                break
            query_chroma(chroma, user_input)


if __name__ == "__main__":
    main()
