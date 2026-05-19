"""
Retriever function for knowledge

"""



import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


# convert text into vectors (numbers) so it can be stored and searched in Neo4j
# loads the all-MiniLM-L6-v2 model
# converts a list of text chunks into vectors (used when storing)
# converts a single search query into a vector (used when searching)
# _normalize part just makes sure all vectors are on the same scale so comparisons are fair
class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [self._normalize(v) for v in vecs]

    def embed_query(self, text):
        vec = self._model.encode([text], convert_to_numpy=True)[0]
        return self._normalize(vec)

    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else np.zeros_like(v).tolist()

# Gets all the info into one spot so you can congifure it easily with build_retriever()
@dataclass
class RetrieverConfig:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    index_name: str = "essay_chunkAgentSpace"
    fulltext_index_name: str = "essay_chunkAgentSpace"
    node_label: str = "Chunk"
    text_property: str = "text"
    embedding_property: str = "embedding"
    k: int = 4
    context_size: int = 1
    vector_weight: float = 2.0
    bm25_weight: float = 1.0
    embedding_model: str = "all-MiniLM-L6-v2"

#
@dataclass
class Retriever:
    config: RetrieverConfig
    graph: object
    vectorstore: object
    embeddings: object


def build_retriever(
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    **kwargs,
) -> Retriever:
    config = RetrieverConfig(
        neo4j_uri=neo4j_uri or os.environ["NEO4J_URI"],
        neo4j_username=neo4j_username or os.environ["NEO4J_USERNAME"],
        neo4j_password=neo4j_password or os.environ["NEO4J_PASSWORD"],
        neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        **kwargs,
    )
    embeddings = MiniLMEmbeddings(config.embedding_model)
    graph = Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )
    vectorstore = Neo4jVector(
        embedding=embeddings,
        url=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        index_name=config.index_name,
        node_label=config.node_label,
        text_node_property=config.text_property,
        embedding_node_property=config.embedding_property,
        database=config.neo4j_database,
    )
    return Retriever(config=config, graph=graph, vectorstore=vectorstore, embeddings=embeddings)


def _get_chunk_context(graph: Neo4jGraph, chunk_id: str, context_size: int = 1):
    cypher = """
    MATCH (c:Chunk)
    WHERE c.chunk_id = $chunk_id
    OPTIONAL MATCH (before:Chunk)-[:NEXT*1..%d]->(c)
    WITH c, before ORDER BY before.chunk_id ASC
    WITH c, collect(before.text) AS before_texts
    OPTIONAL MATCH (c)-[:NEXT*1..%d]->(after:Chunk)
    WITH c, before_texts, after ORDER BY after.chunk_id ASC
    WITH c, before_texts, collect(after.text) AS after_texts
    RETURN
        c.text     AS current_text,
        c.chunk_id AS current_id,
        before_texts,
        after_texts
    """ % (context_size, context_size)
    result = graph.query(cypher, params={"chunk_id": chunk_id})
    return result[0] if result else None


def _vector_search(retriever: Retriever, query_text: str, k: int, context_size: int):
    raw = retriever.vectorstore.similarity_search_with_score(query_text, k=k)
    results = []
    for doc, score in raw:
        cid = doc.metadata.get("chunk_id")
        results.append({
            "text": doc.page_content,
            "score": float(score),
            "chunk_id": cid,
            "source": doc.metadata.get("source", "unknown"),
            "context": _get_chunk_context(retriever.graph, cid, context_size) if cid is not None else None,
        })
    return results


def _bm25_search(retriever: Retriever, query_text: str, k: int, context_size: int):
    cypher = """
    CALL db.index.fulltext.queryNodes($index, $query)
    YIELD node, score
    RETURN node.text AS text, node.chunk_id AS chunk_id, node.source AS source, score
    ORDER BY score DESC LIMIT $k
    """
    raw = retriever.graph.query(
        cypher,
        params={"index": retriever.config.fulltext_index_name, "query": query_text, "k": k},
    )
    results = []
    for r in raw:
        cid = r["chunk_id"]
        results.append({
            "text": r["text"],
            "score": float(r["score"]),
            "chunk_id": cid,
            "source": r.get("source", "unknown"),
            "context": _get_chunk_context(retriever.graph, cid, context_size) if cid is not None else None,
        })
    return results


def _reciprocal_rank_fusion(vector_results, bm25_results, k=60, vector_weight=2.0, bm25_weight=1.0):
    scores = {}
    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        if cid is None:
            continue
        scores.setdefault(cid, {"score": 0.0, "data": r})
        scores[cid]["score"] += vector_weight / (k + rank + 1)
    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        if cid is None:
            continue
        scores.setdefault(cid, {"score": 0.0, "data": r})
        scores[cid]["score"] += bm25_weight / (k + rank + 1)
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        {
            "chunk_id": r["data"]["chunk_id"],
            "text": r["data"]["text"],
            "source": r["data"]["source"],
            "rrf_score": r["score"],
            "context": r["data"]["context"],
        }
        for r in ranked
    ]


def _build_context_string(result: dict) -> str:
    ctx = result.get("context") or {}
    parts = []
    before = " ".join(ctx.get("before_texts", []))
    if before:
        parts.append(before)
    parts.append(f"[CURRENT] {result['text']} [END]")
    after = " ".join(ctx.get("after_texts", []))
    if after:
        parts.append(after)
    return " ".join(parts)


def retrieve(
    retriever: Retriever,
    query: str,
    k: Optional[int] = None,
    context_size: Optional[int] = None,
    vector_weight: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    include_context_string: bool = True,
) -> list[dict]:
    cfg = retriever.config
    k             = k             if k             is not None else cfg.k
    context_size  = context_size  if context_size  is not None else cfg.context_size
    vector_weight = vector_weight if vector_weight is not None else cfg.vector_weight
    bm25_weight   = bm25_weight   if bm25_weight   is not None else cfg.bm25_weight

    v_results = _vector_search(retriever, query, k=k * 2, context_size=context_size)
    b_results = _bm25_search(retriever, query, k=k * 2, context_size=context_size)
    fused = _reciprocal_rank_fusion(v_results, b_results, vector_weight=vector_weight, bm25_weight=bm25_weight)
    top = fused[:k]

    if include_context_string:
        for r in top:
            r["context_string"] = _build_context_string(r)

    return top