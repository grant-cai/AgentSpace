"""
PURPOSE OF THIS FILE:

function returns results as plain Python dict
LangChain's chain doesn't understand dicts. It only understands LangChain Document objects
This code is for converting the dicts into Document objects so LangChain can
work with them
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List, Optional

from .Retriever import build_retriever, retrieve, Retriever


class Neo4jHybridRetriever(BaseRetriever):
    neo4j_retriever: Retriever
    k: int = 4
    context_size: int = 1
    vector_weight: float = 2.0
    bm25_weight: float = 1.0

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        results = retrieve(
            self.neo4j_retriever,
            query=query,
            k=self.k,
            context_size=self.context_size,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            include_context_string=True,
        )
        return [
            Document(
                page_content=r["context_string"] or r["text"],
                metadata={
                    "chunk_id": r["chunk_id"],
                    "source": r["source"],
                    "rrf_score": r["rrf_score"],
                },
            )
            for r in results
        ]

# Calls build_retriever() internally
def make_neo4j_retriever(
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    index_name: str = "essay_chunkAgentSpace",
    k: int = 4,
    **kwargs,
) -> Neo4jHybridRetriever:
    r = build_retriever(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        index_name=index_name,
        **kwargs,
    )
    return Neo4jHybridRetriever(neo4j_retriever=r, k=k)