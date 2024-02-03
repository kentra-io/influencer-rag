from abc import ABC
from enum import Enum

import numpy as np
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain_core.retrievers import BaseRetriever

from app import config
from app.vector_db.chroma_provider import get_chroma
from app.vector_db.elasticsearch_provider import get_elasticsearch
from app.vector_db.weaviate_provider import get_weaviate, get_weaviate_hybrid_retriever, get_weaviate_mrr_retriever


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

    def similarity_search_with_score(self, users_query, k, hybrid_search, alpha, mmr_search):
        if mmr_search:
            if self.vector_db_type != VectorDbType.WEAVIATE:
                raise NotImplementedError("MMR search is only supported for Weaviate")
            if hybrid_search:
                raise NotImplementedError("MMR search not supported together with hybrid search")
            k_to_fetch = k + config.additional_items_to_fetch_for_mmr
        else:
            k_to_fetch = k

        if hybrid_search:
            if self.vector_db_type != VectorDbType.WEAVIATE:
                raise NotImplementedError("Hybrid search is only supported for Weaviate")
            # Retriever is created every time as k needs to be passed to the retriever
            retriever = get_weaviate_hybrid_retriever(k_to_fetch, alpha)
            results_with_score_in_metadata = retriever.get_relevant_documents(users_query, score=True)
            unfiltered_results = [
                (result, float(result.metadata['_additional']['score'])) for result in results_with_score_in_metadata
            ]
        else:
            unfiltered_results = self.vector_store.similarity_search_with_score(users_query, k_to_fetch)

        if mmr_search:
            results = self.apply_mrr_filter(unfiltered_results, users_query, k)
        else:
            results = unfiltered_results

        # TODO That is a temporary solution!
        #  It seems that ES returns cosine distance while Chroma returns cosine similarity, so for now
        #  I just flip the ES scores. We need investigate in the ES documentation.
        if self.vector_db_type == VectorDbType.ELASTICSEARCH:
            normalized_results = [(document, 1 - score) for document, score in results]
        else:
            normalized_results = results

        return normalized_results

    def apply_mrr_filter(self, results, query, k):
        query_embedding = self.vector_store.embeddings.embed_query(query)
        embeddings = [result[0].metadata["_additional"]["vector"] for result in results]
        mmr_selected = maximal_marginal_relevance(np.array(query_embedding), embeddings, k=k)
        mmr_selected_docs = [results[idx] for idx in mmr_selected]
        return mmr_selected_docs

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
