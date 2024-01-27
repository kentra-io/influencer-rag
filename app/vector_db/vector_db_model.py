from abc import ABC
from enum import Enum

from langchain.schema.vectorstore import VectorStore
from langchain_core.retrievers import BaseRetriever

from app.vector_db.chroma_provider import get_chroma
from app.vector_db.elasticsearch_provider import get_elasticsearch
from app.vector_db.weaviate_provider import get_weaviate, get_weaviate_hybrid_retriever


class VectorDbType(Enum):
    WEAVIATE = "Weaviate"
    ELASTICSEARCH = "ElasticSearch"
    CHROMA = "Chroma"


class VectorDb(ABC):
    vector_db_type: VectorDbType
    vector_store: VectorStore
    retriever: BaseRetriever

    def __init__(self, vector_db_type: VectorDbType, vector_store: VectorStore, retriever: BaseRetriever = None):
        self.vector_db_type = vector_db_type
        self.vector_store = vector_store
        self.retriever = retriever

    def add_texts(self, texts, metadatas):
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search_with_score(self, users_query, k, hybrid_search, alpha):
        if hybrid_search:
            if self.vector_db_type != VectorDbType.WEAVIATE:
                raise NotImplementedError("Hybrid search is not supported for Weaviate")
            # Retriever is created every time as k needs to be passed to the retriever
            retriever = get_weaviate_hybrid_retriever(k, alpha)
            results_with_score_in_metadata = retriever.get_relevant_documents(users_query, score=True)
            results = [
                (result, float(result.metadata['_additional']['score'])) for result in results_with_score_in_metadata
            ]
        else:
            results = self.vector_store.similarity_search_with_score(users_query, k)

        # TODO That is a temporary solution!
        #  It seems that ES returns cosine distance while Chroma returns cosine similarity, so for now
        #  I just flip the ES scores. We need investigate in the ES documentation.
        if self.vector_db_type == VectorDbType.ELASTICSEARCH:
            normalized_results = [(document, 1 - score) for document, score in results]
        else:
            normalized_results = results

        return normalized_results

    def persist(self):
        if self.vector_db_type == VectorDbType.CHROMA:
            self.vector_store.persist()


def get_vector_db(vector_db: VectorDbType) -> VectorDb:
    if vector_db == VectorDbType.ELASTICSEARCH:
        return VectorDb(VectorDbType.ELASTICSEARCH, get_elasticsearch())
    elif vector_db == VectorDbType.CHROMA:
        return VectorDb(VectorDbType.CHROMA, get_chroma())
    elif vector_db == VectorDbType.WEAVIATE:
        return VectorDb(VectorDbType.WEAVIATE, get_weaviate())
