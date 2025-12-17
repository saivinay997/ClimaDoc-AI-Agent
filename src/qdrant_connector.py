import os
from typing import List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import google.generativeai as genai
from langchain_core.documents import Document

from secrets_loader import get_secret

QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
qdrant_client = QdrantClient(
    url="https://d14b12c7-d670-4b28-91ad-5198f84e092b.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_API_KEY,
)
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "models/text-embedding-004"

# Global variable to store the API key (configured by Streamlit app)
_google_api_key = None

def configure_genai_api_key(api_key: str):
    """Configure the Google Generative AI API key globally."""
    global _google_api_key
    _google_api_key = api_key
    if api_key:
        genai.configure(api_key=api_key)

def create_collection(vector_size: int):
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def store_embeddings_qdrant(
    embedded_chunks: List[dict],
    batch_size: int = 100,
):
    """
    Store Gemini embeddings in Qdrant.

    embedded_chunks item format:
    {
        "embedding": List[float],
        "text": str,
        "metadata": dict
    }
    """

    points = []

    for chunk in embedded_chunks:
        metadata = chunk.get("metadata", {})

        point_id = metadata.get("chunk_id") or str(uuid4())

        payload = {
            "text": chunk["text"],
            **metadata,
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload,
            )
        )

        if len(points) >= batch_size:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            points = []

    # flush remaining points
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )


def embed_query(query: str, api_key: str = None) -> list[float]:
    """
    Generate embedding for a query string.
    
    Args:
        query (str): The query string to embed
        api_key (str, optional): Google API key. If not provided, uses globally configured key.
    
    Returns:
        list[float]: The embedding vector
    """
    # Configure API key if provided
    if api_key:
        genai.configure(api_key=api_key)
    elif not _google_api_key:
        raise ValueError("Google API key must be configured. Call configure_genai_api_key() or pass api_key parameter.")
    
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return response["embedding"]

def retrieve_similar_documents(
    query: str,
    top_k: int = 5,
    metadata_filter: Optional[dict] = None,
    api_key: str = None,
) -> List[Document]:
    """
    Retrieve similar documents from Qdrant using Gemini embeddings.
    
    Args:
        query (str): The search query
        top_k (int): Number of documents to retrieve
        metadata_filter (Optional[dict]): Optional metadata filter
        api_key (str, optional): Google API key for embeddings. If not provided, 
                                uses globally configured key.

    Returns:
        List[Document]: List of similar documents
    """

    qdrant_filter = None
    if metadata_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in metadata_filter.items()
            ]
        )
    query_embedding = embed_query(query, api_key=api_key)
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    documents: List[Document] = []

    for point in results.points:
        payload = point.payload or {}

        documents.append(
            Document(
                page_content=payload.get("text", ""),
                metadata={
                    **payload,
                    "score": point.score,
                },
            )
        )

    return documents
