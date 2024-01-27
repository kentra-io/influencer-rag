from langchain.embeddings import SentenceTransformerEmbeddings
import weaviate
from langchain.vectorstores.weaviate import Weaviate
# from langchain_community.vectorstores import Weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

weaviate_embedding_model = "all-MiniLM-L6-v2"
weaviate_index_name = "youtube_transcripts"

client = weaviate.Client(
    url="http://localhost:8080"
)

def get_weaviate():

    # vectorstore = WeaviateHybridSearchRetriever(
    vectorstore = Weaviate(
        client=client,
        index_name="LangChain",
        text_key="text",
        embedding=SentenceTransformerEmbeddings(model_name=weaviate_embedding_model),
        attributes=["title", "channel", "url",  "file", "paragraph"]
    )

    return vectorstore

def get_weaviate_hybrid_retriever(k, alpha):
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="LangChain",
        text_key="text",
        k=k,
        alpha=alpha,
        embedding=SentenceTransformerEmbeddings(model_name=weaviate_embedding_model),
        attributes=["title", "channel", "url",  "file", "paragraph"]
    )

    return retriever
