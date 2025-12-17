import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

import rag_prompts
from rag_pipelines import document_retrieval_pipeline


model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def rag_answer_generator(query: str, top_k: int = 5) -> dict:
    """
    End-to-end RAG pipeline with Gemini.
    """
    query = HumanMessage(content=query)
    

    # Step 1: Retrieve documents
    context, sources, documents = document_retrieval_pipeline(query.content, top_k)
    print(f"Retireved the context.")
    if not context:
        answer = model.invoke([query])
        return {
            "answer": answer.content,
            "rag_used": False,
            "sources": [],
        }


    # Step 2: Build RAG prompt
    rag_prompt = [SystemMessage(content=rag_prompts.rag_answer_prompt)]
    query_with_context = [HumanMessage(content=f"Query: {query.content}\n\nContext: {context}")]

    # Step 6: Generate answer
    answer = model.invoke(rag_prompt + query_with_context)

    return {
        "answer": answer,
        "rag_used": True,
        "source": sources
    }
