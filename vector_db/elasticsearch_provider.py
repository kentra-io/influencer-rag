from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import ElasticsearchStore

elasticsearch_embedding_model = "all-MiniLM-L6-v2"
elasticsearch_index_name = "youtube_transcripts"


def get_elasticsearch():
    elastic_vector_search = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=elasticsearch_index_name,
        embedding=SentenceTransformerEmbeddings(model_name=elasticsearch_embedding_model)
    )

    return elastic_vector_search
