import sys
import time
from datetime import timedelta

from vector_db import chroma_provider

chroma = chroma_provider.get_chroma()

if len(sys.argv) > 1:
    print(f"Question: '{sys.argv[1]}'\n")

    start_time = time.time()
    results = chroma.similarity_search_with_score(sys.argv[1], 5)
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
else:
    print("Please provide your prompt in the first parameter")
