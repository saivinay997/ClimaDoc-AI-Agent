import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from typing import Optional

import prompts
from rag_pipelines import document_retrieval_pipeline

# Global model instance (lazy initialization)
_model = None

def get_model(api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
    """
    Get or create the model instance. Uses globally configured API key if not provided.
    
    Args:
        api_key: Optional API key. If not provided, tries to use globally configured key.
        model_name: Model name to use
    
    Returns:
        ChatGoogleGenerativeAI instance
    """
    global _model
    if _model is None or api_key:
        if api_key:
            _model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7
            )
        else:
            # Try to create without explicit API key (will use environment or global config)
            try:
                _model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.7
                )
            except Exception as e:
                raise ValueError(
                    "API key is required. Please configure it via configure_genai_api_key() "
                    f"or pass it as a parameter. Error: {str(e)}"
                )
    return _model


def rag_answer_generator(
    query: str, 
    top_k: int = 5, 
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash"
) -> dict:
    """
    End-to-end RAG pipeline with Gemini.
    
    Args:
        query: The user query
        top_k: Number of documents to retrieve
        api_key: Optional API key. If not provided, uses globally configured key.
        model_name: Model name to use
    
    Returns:
        dict with answer, rag_used flag, and sources
    """
    query = HumanMessage(content=query)
    
    # Get model instance
    model = get_model(api_key=api_key, model_name=model_name)

    # Step 1: Retrieve documents
    context, sources, documents = document_retrieval_pipeline(
        query=query.content, 
        top_k=top_k,
        api_key=api_key
    )
    print(f"Retrieved the context.")
    
    if not context:
        answer = model.invoke([query])
        return {
            "answer": answer.content,
            "rag_used": False,
            "sources": [],
        }

    # Step 2: Build RAG prompt
    rag_prompt = [SystemMessage(content=prompts.rag_answer_prompt)]
    query_with_context = [HumanMessage(content=f"Query: {query.content}\n\nContext: {context}")]

    # Step 3: Generate answer
    answer = model.invoke(rag_prompt + query_with_context)

    return {
        "answer": answer,
        "rag_used": True,
        "source": sources
    }
