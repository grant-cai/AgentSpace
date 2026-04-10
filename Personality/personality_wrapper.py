"""
Personality Wrapper

Exposes personality data in two tiers:
  - get_profile(): structured JSON profile, always available as baseline context
  - retrieve(): FAISS-backed interview retrieval, called on demand when deeper context is needed
"""

import json
import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class PersonalityWrapper:
    def __init__(
        self,
        profile_path: str = "personality_summary.json",
        faiss_dir: str = "./faiss_db",
    ):
        with open(profile_path, "r", encoding="utf-8") as f:
            self._profile = json.load(f)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = FAISS.load_local(
            faiss_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def get_profile(self) -> dict:
        """
        Returns the full personality profile as a dict.
        The synthesizer should always include this as baseline context.
        """
        return self._profile

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieves the k most relevant interview chunks for the given query.
        Called on demand by the orchestrator when the JSON profile is insufficient.

        Returns a list of chunk text strings.
        """
        results = self._vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
