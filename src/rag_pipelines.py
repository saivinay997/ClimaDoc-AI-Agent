import os
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document

from qdrant_connector import create_collection, store_embeddings_qdrant, retrieve_similar_documents

EMBEDDING_MODEL = "models/text-embedding-004"

# Global variable to store the API key (configured by Streamlit app)
_google_api_key = None

def configure_genai_api_key(api_key: str):
    """Configure the Google Generative AI API key globally."""
    global _google_api_key
    _google_api_key = api_key
    if api_key:
        genai.configure(api_key=api_key)


def langchain_document_loader(
    base_dir: Path,
    show_progress: bool = True,
) -> List[Document]:
    """
    Create document loaders for TXT, PDF, CSV and DOCX files using LangChain.

    Args:
        base_dir (Path): Directory containing uploaded documents
        show_progress (bool): Show loading progress

    Returns:
        List[Document]: Loaded LangChain documents
    """

    loaders_config = [
        {
            "glob": "**/*.txt",
            "loader_cls": TextLoader,
            "loader_kwargs": {"encoding": "utf-8"},
            "file_type": "txt",
        },
        {
            "glob": "**/*.pdf",
            "loader_cls": PyPDFLoader,
            "loader_kwargs": {},
            "file_type": "pdf",
        },
        {
            "glob": "**/*.csv",
            "loader_cls": CSVLoader,
            "loader_kwargs": {"encoding": "utf-8"},
            "file_type": "csv",
        },
        {
            "glob": "**/*.docx",
            "loader_cls": Docx2txtLoader,
            "loader_kwargs": {},
            "file_type": "docx",
        },
    ]

    documents: List[Document] = []
    base_dir = Path(base_dir)
    for config in loaders_config:
        try:
            loader = DirectoryLoader(
                base_dir.as_posix(),
                glob=config["glob"],
                loader_cls=config["loader_cls"],
                loader_kwargs=config["loader_kwargs"],
                show_progress=show_progress,
            )

            docs = loader.load()

            # enrich metadata
            for doc in docs:
                doc.metadata.update(
                    {
                        "file_type": config["file_type"],
                        "source_dir": str(base_dir),
                    }
                )

            documents.extend(docs)

        except Exception as e:
            print(f"⚠️ Failed to load {config['glob']}: {e}")

    return documents

def chunk_langchain_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Chunk LangChain Documents while preserving and enriching metadata.

    Args:
        documents (List[Document]): Loaded LangChain documents
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        List[Document]: Chunked documents
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunked_docs: List[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)

        for idx, split in enumerate(splits):
            chunked_docs.append(
                Document(
                    page_content=split.replace("-\n", "").replace("\n", " "),
                    metadata={
                        **doc.metadata,
                        "chunk_id": str(uuid4()),
                        "chunk_index": idx,
                        "chunk_size": len(split),
                    },
                )
            )

    return chunked_docs


def generate_gemini_embeddings(
    documents: List[Document],
    batch_size: int = 16,
    sleep_seconds: float = 0.1,
    api_key: str = None,
) -> List[dict]:
    """
    Generate Gemini embeddings for LangChain Documents.

    Args:
        documents (List[Document]): Documents to generate embeddings for
        batch_size (int): Number of documents to process in each batch
        sleep_seconds (float): Sleep time between API calls
        api_key (str, optional): Google API key. If not provided, uses globally configured key.

    Returns a list of dicts:
    {
        "embedding": List[float],
        "text": str,
        "metadata": dict
    }
    """
    # Configure API key if provided
    if api_key:
        genai.configure(api_key=api_key)
    elif not _google_api_key:
        raise ValueError("Google API key must be configured. Call configure_genai_api_key() or pass api_key parameter.")

    embedded_chunks = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        for doc in batch:
            try:
                response = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=doc.page_content,
                    task_type="retrieval_document",
                )

                embedded_chunks.append(
                    {
                        "embedding": response["embedding"],
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )

            except Exception as e:
                print(
                    f"⚠️ Embedding failed "
                    f"(source={doc.metadata.get('source')}): {e}"
                )

            time.sleep(sleep_seconds)

    return embedded_chunks


def document_ingestion_pipeline(base_dir: str, api_key: str = None) -> dict:
    """
    Complete document ingestion pipeline that loads, chunks, embeds, and stores documents.
    
    This function orchestrates the entire document ingestion process:
    1. Loads documents from the specified directory (supports PDF, TXT, CSV, DOCX)
    2. Chunks documents into smaller, manageable pieces
    3. Generates embeddings using Google Gemini embedding model
    4. Creates/updates the Qdrant collection
    5. Stores embeddings in the vector database
    
    Args:
        base_dir (str): Path to the directory containing documents to ingest.
                        Can be a string path or Path object.
        api_key (str, optional): Google API key for embeddings. If not provided, 
                                uses globally configured key via configure_genai_api_key().
    
    Returns:
        dict: Summary dictionary containing:
            - "status" (str): "success" or "error"
            - "documents_loaded" (int): Number of documents loaded
            - "chunks_created" (int): Number of chunks created
            - "embeddings_generated" (int): Number of embeddings generated
            - "embeddings_stored" (int): Number of embeddings stored in Qdrant
            - "message" (str): Status message or error description
    
    Raises:
        ValueError: If base_dir is empty or invalid, or if API key is not configured
        FileNotFoundError: If the directory doesn't exist
        Exception: For errors during embedding generation or storage
    
    Example:
        >>> result = document_ingestion_pipeline("./documents", api_key="your-api-key")
        >>> print(result["embeddings_stored"])
        150
    """
    try:
        # Validate input
        if not base_dir or not str(base_dir).strip():
            raise ValueError("base_dir cannot be empty")
        
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        # Step 1: Load documents
        documents = langchain_document_loader(base_dir=base_path)
        if not documents:
            return {
                "status": "warning",
                "documents_loaded": 0,
                "chunks_created": 0,
                "embeddings_generated": 0,
                "embeddings_stored": 0,
                "message": f"No documents found in {base_dir}"
            }
        
        # Step 2: Chunk documents
        documents_chunked = chunk_langchain_documents(documents)
        print(f"✓ Created {len(documents_chunked)} chunks from {len(documents)} documents")
        
        # Step 3: Generate embeddings
        embeddings = generate_gemini_embeddings(documents_chunked, api_key=api_key)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        if not embeddings:
            return {
                "status": "error",
                "documents_loaded": len(documents),
                "chunks_created": len(documents_chunked),
                "embeddings_generated": 0,
                "embeddings_stored": 0,
                "message": "Failed to generate embeddings"
            }
        
        # Step 4: Create/update collection
        create_collection(vector_size=768)
        print("✓ Collection ready")
        
        # Step 5: Store embeddings
        store_embeddings_qdrant(embedded_chunks=embeddings)
        print(f"✓ Stored {len(embeddings)} embeddings in Qdrant DB")
        
        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(documents_chunked),
            "embeddings_generated": len(embeddings),
            "embeddings_stored": len(embeddings),
            "message": f"Successfully ingested {len(embeddings)} embeddings from {len(documents)} documents"
        }
        
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "documents_loaded": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "message": error_msg
        }
    except FileNotFoundError as e:
        error_msg = f"Directory not found: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "documents_loaded": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error during ingestion: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "documents_loaded": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "message": error_msg
        }


def document_retrieval_pipeline(query: str, top_k: int = 5, api_key: str = None) -> Tuple[str, List[str], List[Document]]:
    """
    Retrieve relevant documents from the vector database based on a query.
    
    This function performs semantic search to find the most relevant document chunks
    for a given query, formats them into a context string, and extracts unique sources.
    
    Args:
        query (str): The search query/question to find relevant documents for.
        top_k (int, optional): Maximum number of similar documents to retrieve. 
                              Defaults to 5. Must be a positive integer.
        api_key (str, optional): Google API key for embeddings. If not provided, 
                                uses globally configured key via configure_genai_api_key().
    
    Returns:
        tuple[str, List[str], List[Document]]: A tuple containing:
            - context (str): Formatted string containing all retrieved document chunks,
                           with each chunk prefixed by "- " and separated by newlines.
                           Empty string if no documents found.
            - sources (List[str]): List of unique source file names (without path).
                                 Empty list if no documents found.
            - documents (List[Document]): List of LangChain Document objects retrieved.
                                        Empty list if no documents found.
    
    Raises:
        ValueError: If query is empty or top_k is not a positive integer.
        Exception: For errors during document retrieval from Qdrant.
    
    Example:
        >>> context, sources, docs = document_retrieval_pipeline("What is climate change?", top_k=3, api_key="your-key")
        >>> print(f"Found {len(sources)} sources")
        Found 2 sources
        >>> print(len(context))
        1250
    """
    try:
        # Validate input
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        
        # Retrieve similar documents
        documents = retrieve_similar_documents(query=query.strip(), top_k=top_k, api_key=api_key)
        
        if not documents:
            return "", [], []
        
        # Build context string and extract sources
        context_parts = []
        sources = []
        seen_sources = set()
        
        for document in documents:
            # Add document content to context
            if document.page_content:
                context_parts.append(f"- {document.page_content.strip()}")
            
            # Extract and deduplicate source file names
            source_path = document.metadata.get("source", "")
            if source_path:
                # Handle both Windows (\) and Unix (/) path separators
                source_name = source_path.replace("\\", "/").split("/")[-1]
                if source_name and source_name not in seen_sources:
                    sources.append(source_name)
                    seen_sources.add(source_name)
        
        # Join context parts with newlines
        context = "\n ".join(context_parts) if context_parts else ""
        
        return context, sources, documents
        
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        print(f"❌ {error_msg}")
        return "", [], []
    except Exception as e:
        error_msg = f"Error during document retrieval: {str(e)}"
        print(f"❌ {error_msg}")
        return "", [], []



    