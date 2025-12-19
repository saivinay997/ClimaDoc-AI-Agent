import os
from typing import List, Optional
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
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
google_api_key = None

def configure_genai_api_key(api_key: str):
    """Configure the Google Generative AI API key globally."""
    global google_api_key
    google_api_key = api_key
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
    elif os.getenv("GOOGLE_API_KEY") is not None:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    elif not google_api_key:
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


def delete_all_records_from_collection() -> dict:
    """
    Delete all records from the Qdrant collection.
    
    Returns:
        dict: Status dictionary with 'success' (bool) and 'message' (str) keys
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(
            col.name == COLLECTION_NAME for col in collections.collections
        )
        
        if not collection_exists:
            return {
                "success": False,
                "message": f"Collection '{COLLECTION_NAME}' does not exist.",
                "deleted_count": 0
            }
        
        # Scroll through all points to get their IDs
        point_ids = []
        offset = None
        
        while True:
            # Scroll to get points (limit of 100 per batch)
            result = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            
            batch_ids = [point.id for point in result[0]]
            if not batch_ids:
                break
                
            point_ids.extend(batch_ids)
            offset = result[1]  # Next offset
            
            # If no next offset, we've reached the end
            if offset is None:
                break
        
        # Delete all points if any exist
        if point_ids:
            # Delete in batches to avoid potential issues with large collections
            batch_size = 100
            for i in range(0, len(point_ids), batch_size):
                batch = point_ids[i:i + batch_size]
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=PointIdsList(points=batch),
                    wait=True
                )
            return {
                "success": True,
                "message": f"Successfully deleted {len(point_ids)} records from collection '{COLLECTION_NAME}'.",
                "deleted_count": len(point_ids)
            }
        else:
            return {
                "success": True,
                "message": f"Collection '{COLLECTION_NAME}' is already empty.",
                "deleted_count": 0
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Error deleting records: {str(e)}",
            "deleted_count": 0
        }
