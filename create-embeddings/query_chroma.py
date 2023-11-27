import time
from datetime import timedelta

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

chroma = Chroma(persist_directory="../chroma_db", collection_name="youtube_transcripts",
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                )
question = "what is Artificial General Intelligence?"

print(f"Question: '{question}'\n")

start_time = time.time()
results = chroma.similarity_search(question)
execution_time = time.time() - start_time

print(f"Generated the following answers in {timedelta(seconds=execution_time)}:\n")

for result in results:
    print(f"Reply: '{result.page_content}'")
    print(f"Metadata: '{result.metadata}'\n")
