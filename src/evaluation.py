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
from dotenv import load_dotenv
load_dotenv()
# Environment variables required:
# - LANGSMITH_API_KEY
# - LANGSMITH_PROJECT
# - GEMINI_API_KEY (or GOOGLE_API_KEY)

# Enable LangSmith tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")


class EvaluationOutput(BaseModel):
    """Structured output for evaluation results."""
    verdict: str = Field(description="'pass' or 'fail'")
    score: float = Field(description="Numeric score from 0 to 1")
    is_hallucinated: bool = Field(description="Whether the answer contains hallucinated information")
    is_grounded: bool = Field(description="Whether the answer is grounded in provided context")
    is_complete: bool = Field(description="Whether the answer is complete and actionable")
    feedback: str = Field(description="Concise, actionable feedback")


JUDGE_PROMPT = """You are an expert evaluator for the ClimaDoc AI agent, which provides weather information and document-based answers.

Your task is to evaluate the quality of the agent's final answer. You must NOT generate answers—only evaluate them.
Note: The ClimaDoc AI agent used ###real-time weather data### to answer any of the weather related queries.
    - Even if your data is December 2, 2023, You are currently in the future. So the current year now is 2025 and 2026.
## Evaluation Criteria

1. **Factual Accuracy**: Is the information correct? Penalize hallucinated weather data heavily.
2. **Groundedness**: For document-based answers, is the response grounded in the provided context? Penalize ungrounded claims.
3. **Completeness**: Does the answer address all parts of the user's query?
4. **Actionability**: Is the answer useful and actionable for the user?

## Scoring Guidelines

- **1.0**: Perfect answer - accurate, complete, well-grounded, actionable
- **0.8-0.9**: Good answer - minor issues that don't affect usefulness
- **0.6-0.7**: Acceptable - some missing info or minor inaccuracies
- **0.4-0.5**: Poor - significant issues but partially useful
- **0.0-0.3**: Fail - hallucinations, factual errors, or completely off-topic

## ClimaDoc-Specific Rules

- FAIL for hallucinated weather data (made-up temperatures, conditions, etc.). Reminding you that real time Weather data is injected through context.
- FAIL for document answers not grounded in provided context
- ALLOW minor phrasing or formatting issues
- FAIL only for: factual errors, missing critical information, or tool misuse

## Input Format

User Query: {user_query}

Final Answer: {final_answer}

Context (if available): {context}

Evaluate the answer and provide your structured assessment."""


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
    
    prompt = JUDGE_PROMPT.format(
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


def run_batch_evaluation(examples: list[dict]) -> list[dict]:
    """
    Run evaluation on multiple examples.
    
    Args:
        examples: List of dicts with 'user_query', 'final_answer', and optional 'context'
    
    Returns:
        List of evaluation results
    """
    results = []
    for ex in examples:
        result = evaluate_answer(
            user_query=ex["user_query"],
            final_answer=ex["final_answer"],
            context=ex.get("context")
        )
        results.append(result)
    return results


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
