import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from uuid import uuid4
import time
load_dotenv()

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

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

EMBEDDING_MODEL = "models/text-embedding-004"


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
) -> List[dict]:
    """
    Generate Gemini embeddings for LangChain Documents.

    Returns a list of dicts:
    {
        "embedding": List[float],
        "text": str,
        "metadata": dict
    }
    """

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


def document_ingestion_pipeline(base_dir: str):
    
    documents = langchain_document_loader(base_dir=base_dir)
    documents_chunked = chunk_langchain_documents(documents)
    embeddings = generate_gemini_embeddings(documents_chunked)
    create_collection(vector_size=768)
    store_embeddings_qdrant(embedded_chunks=embeddings)
    print(f"Injested {len(embeddings)} embeddings in Qdrant DB")


def document_retrival_pipeline(query:str, top_k:int=5):
    
    documents = retrieve_similar_documents(query=query, top_k=top_k)
    
    context = ""
    sources = []
    for document in documents:
        context+="\n " + f"- {document.page_content}"
        source = document.metadata.get("source").split("\\")[-1]
        if source not in sources:
            sources.append(source)
    
    return context, sources, documents



    