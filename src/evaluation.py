"""
LangSmith Evaluation Module for ClimaDoc Agent
Uses Gemini as a judge model to evaluate final answers.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from langsmith import Client, traceable
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import judge_prompt_langsmith, judge_prompt_langsmith
from dotenv import load_dotenv
load_dotenv()
from secrets_loader import get_secret
# Environment variables required:
# - LANGSMITH_API_KEY
# - LANGSMITH_PROJECT
# - GEMINI_API_KEY (or GOOGLE_API_KEY)

# Enable LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", get_secret("LANGSMITH_ENDPOINT"))
os.environ.setdefault("LANGSMITH_API_KEY", get_secret("LANGSMITH_API_KEY"))


class EvaluationOutput(BaseModel):
    """Structured output for evaluation results."""
    verdict: str = Field(description="'pass' or 'fail'")
    score: float = Field(description="Numeric score from 0 to 1")
    is_hallucinated: bool = Field(description="Whether the answer contains hallucinated information")
    is_grounded: bool = Field(description="Whether the answer is grounded in provided context")
    is_complete: bool = Field(description="Whether the answer is complete and actionable")
    feedback: str = Field(description="Concise, actionable feedback")


def get_judge_llm():
    """Initialize Gemini as the judge model."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.0,
    )


@traceable(name="evaluate_climadoc_answer")
def evaluate_answer(
    user_query: str,
    final_answer: str,
    context: Optional[str] = None
) -> dict:
    """
    Evaluate a ClimaDoc agent answer using Gemini as judge.
    
    Args:
        user_query: The original user question
        final_answer: The agent's final answer
        context: Optional retrieved context or tool outputs
    
    Returns:
        Dictionary with verdict, score, and feedback
    """
    llm = get_judge_llm()
    judge_llm = llm.with_structured_output(EvaluationOutput)
    
    prompt = judge_prompt_langsmith.format(
        user_query=user_query,
        final_answer=final_answer,
        context=context or "No context provided"
    )
    
    response: EvaluationOutput = judge_llm.invoke([
        SystemMessage(content="You are an evaluation judge. Provide structured assessment only."),
        HumanMessage(content=prompt)
    ])
    
    return {
        "verdict": response.verdict,
        "score": response.score,
        "is_hallucinated": response.is_hallucinated,
        "is_grounded": response.is_grounded,
        "is_complete": response.is_complete,
        "feedback": response.feedback,
    }


class ClimaDocEvaluator(RunEvaluator):
    """LangSmith RunEvaluator for ClimaDoc agent outputs."""
    
    def __init__(self):
        self.llm = get_judge_llm().with_structured_output(EvaluationOutput)
    
    def evaluate_run(self, run, example=None) -> EvaluationResult:
        """Evaluate a LangSmith run."""
        # Extract inputs and outputs from the run
        user_query = ""
        final_answer = ""
        context = ""
        
        if run.inputs:
            user_query = run.inputs.get("query", run.inputs.get("input", ""))
        
        if run.outputs:
            final_answer = run.outputs.get("output", run.outputs.get("answer", str(run.outputs)))
        
        # Get reference from example if available
        reference = ""
        if example and example.outputs:
            reference = example.outputs.get("answer", "")
        
        # Run evaluation
        result = evaluate_answer(user_query, final_answer, context or reference)
        
        return EvaluationResult(
            key="climadoc_quality",
            score=result["score"],
            comment=result["feedback"],
            metadata={
                "verdict": result["verdict"],
                "is_hallucinated": result["is_hallucinated"],
                "is_grounded": result["is_grounded"],
                "is_complete": result["is_complete"],
            }
        )



# LangSmith client for programmatic access
def get_langsmith_client() -> Client:
    """Get LangSmith client instance."""
    return Client()


if __name__ == "__main__":
    # Example usage
    test_result = evaluate_answer(
        user_query="What's the weather in New York?",
        final_answer="The current weather in New York is 45°F with partly cloudy skies.",
        context="API returned: temp=45, condition=partly_cloudy, location=New York"
    )
    print("Evaluation Result:", test_result)
