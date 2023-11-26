# Load or create Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

chroma = Chroma(persist_directory="../chroma_db", collection_name="youtube_transcripts",
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                )

results = chroma.similarity_search("what is Artificial General Intelligence?")

for result in results:
    print(result.page_content)
    print(result.metadata)
