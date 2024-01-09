from abc import ABC
from dataclasses import dataclass
from enum import Enum

from langchain.schema.vectorstore import VectorStore

from vector_db.chroma_provider import get_chroma
from vector_db.elasticsearch_provider import get_elasticsearch

class VectorDbType(Enum):
    ELASTICSEARCH = "ElasticSearch"
    CHROMA = "Chroma"


class VectorDb(ABC):
    vector_db_type: VectorDbType
    vector_store: VectorStore

    def __init__(self, vector_db_type: VectorDbType, vector_store: VectorStore):
        self.vector_db_type = vector_db_type
        self.vector_store = vector_store

    def add_texts(self, texts, metadatas, ids=None):
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search_with_score(self, users_query, k):
        return self.vector_store.similarity_search_with_score(users_query, k)

    def persist(self):
        if (self.vector_db_type == VectorDbType.CHROMA):
            self.vector_store.persist()


def get_vector_db(vector_db: VectorDbType) -> VectorDb:
    if vector_db == VectorDbType.ELASTICSEARCH:
        return VectorDb(VectorDbType.ELASTICSEARCH, get_elasticsearch())
    elif vector_db == VectorDbType.CHROMA:
        return VectorDb(VectorDbType.CHROMA, get_chroma())
