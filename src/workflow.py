import os
from dotenv import load_dotenv
import re
import json

from typing import Annotated, Optional, Sequence, TypedDict, Dict
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI



import prompts
from agent_tools import tools, format_tool_description

load_dotenv()
base_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


class AgentState(TypedDict):
    query: list = []
    action: str = "direct"
    messages: list = []
    num_planning_cycles: int = 0
    is_good_answer: bool = False
    final_answer:str = ""
    tool_responses: list = []
    

class DecisionMakingOutput(BaseModel):
    action: str
    answer: str
    

# Decision making node
def decision_making_node(state: AgentState):
    """
    Enter point of the workflow. Based on the user query, the model can either respond directly or trigger the research workflow.
    """
    #print("#" * 50)
    #print("Decision making node input:", state["messages"][-1])
    
    # Validate input
    if not state["messages"] or not any(msg.content for msg in state["messages"] if hasattr(msg, 'content')):
        return {
            "action": "direct",
            "messages": [AIMessage(content="I apologize, but I didn't receive a valid query. Please try again.")]
        }
    
    try:
        
        decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
        system_prompt = SystemMessage(content=prompts.decision_making_prompt)
        response: DecisionMakingOutput = decision_making_llm.invoke(
            [system_prompt] + state["messages"]
        )
        output = {"action": response.action}
        #print("Decision making node output:", output)
        if response.answer and response.answer != "null":
            # Ensure direct answers are placed in the conversation messages
            output["messages"] = [AIMessage(content=response.answer)]
        print(f"Decision making node: {output}")
        return output
    except Exception as e:
        print(f"Error in decision_making_node: {e}")
        return {
            "action": "direct",
            "messages": [AIMessage(content=f"I encountered an error processing your request: {str(e)}. Please try again.")]
        }
        
        
# Task router function
def router(state: AgentState):
    #print("#" * 50)
    #print("Router node input:", state)
    """Router directing the user query to the appropriate branch of the workflow"""
    if state["action"] != "direct":
        return "planning"
    else:
        return "end"
    
    
# Planning Node
def planning_node(state: AgentState):
    #print("#" * 50)
    #print("Planning node input:", state["messages"][-1])
    """Planning node that creates a step by step plan to answer the user query."""
    # Increment planning cycle counter
    num_planning_cycles = state.get("num_planning_cycles", 0) + 1
    
    # Validate input
    if not state["messages"] or not any(msg.content for msg in state["messages"] if hasattr(msg, 'content')):
        error_message = AIMessage(content="I apologize, but I didn't receive a valid query to plan for. Please try again.")
        return {
            "messages": [error_message],
            "num_planning_cycles": num_planning_cycles
        }
    
    try:
        system_prompt = SystemMessage(
            content=prompts.planning_prompt.format(tools=format_tool_description(tools))
        )
        response = base_llm.invoke([system_prompt] + state["messages"])
        print("Planning node output:", response)
        return {
            "messages": [response],
            "num_planning_cycles": num_planning_cycles
        }
    except Exception as e:
        print(f"Error in planning_node: {e}")
        error_message = AIMessage(content=f"I encountered an error while creating a plan: {str(e)}. Please try again.")
        return {
            "messages": [error_message],
            "num_planning_cycles": num_planning_cycles
        }
        
        
# Agent call node
def agent_node(state: AgentState):
    """Agent call node that uses the LLM with tools to answer the user query."""
    
    
    system_prompt = SystemMessage(content=prompts.agent_prompt)
    
    # Clean the last message content but ensure it's not empty
    last_message = state["messages"][-1]
    if hasattr(last_message, 'content') and last_message.content:
        cleaned_content = re.sub(r"```.*?```", "", last_message.content, flags=re.DOTALL).strip()
        # If cleaning resulted in empty content, keep the original
        if not cleaned_content:
            cleaned_content = last_message.content
        last_message.content = cleaned_content
    
    messages_to_send = [system_prompt] + state["messages"]
    
    # Validate that we have content to send
    if not messages_to_send or not any(msg.content for msg in messages_to_send if hasattr(msg, 'content')):
        error_message = AIMessage(content="I apologize, but I encountered an issue processing your request. Please try rephrasing your question.")
        return {"messages": [error_message]}
    
    try:
        agent_llm = base_llm.bind_tools(tools)
        response = agent_llm.invoke(messages_to_send)
        print("Agent node output:", response)
        return {"messages": [response]}
    except Exception as e:
        print(f"Error in agent_node: {e}")
        error_message = AIMessage(content=f"I encountered an error while processing your request: {str(e)}. Please try again.")
        return {"messages": [error_message]}
    
# Should continue function? after agent resposne
def should_continue(state: AgentState):
    messages=state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Tool call node
def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    tools_dict = {tool.name: tool for tool in tools}
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    
    return {
        "messages":  state["messages"] + outputs,
        "tool_responses": outputs
    }  
    
## answer compiler
def answer_compiler(state:AgentState):
    """Compiles a concise answer by using data from tools output"""
    print(f"\n{state}")
    # Validate input
    if not state["messages"] or not any(msg.content for msg in state["messages"] if hasattr(msg, 'content')):
        error_message = AIMessage(content="I apologize, but I didn't receive a valid query to plan for. Please try again.")
        return {
            "messages": [error_message]
        }
    
    try:
        system_prompt = SystemMessage(
            content=prompts.answer_compiler_prompt
        )
        messages_to_send = [system_prompt] + state["query"] + state["tool_responses"]
        print(f"\n{messages_to_send}")
        response = base_llm.invoke(messages_to_send)
        print("Answer Compiler node output:", response)
        return {
            "messages": [response]
        }
    except Exception as e:
        print(f"Error in answer_compiler: {e}")
        error_message = AIMessage(content=f"I encountered an error while compiling answer: {str(e)}. Please try again.")
        return {
            "messages": [error_message]
        }
    
class JudgeOutput(BaseModel):
    is_good_answer: bool = Field(description="Whether the answer is good or not.")
    feedback: Optional[str] = Field(default=None, description="Detailed feedback about why the answer is not good. It should be None if the answer is good.")
    
def judge_node(state: AgentState):
    """Node to let the LLM judge the quality of its own final answer."""
    #print("#" * 50)
    #print("Judge node input:", state["messages"][-1])
    # End execution if the LLM failed to provide a good answer twice
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        return {"is_good_answer": True}

    # Validate input
    if not state["messages"] or not any(msg.content for msg in state["messages"] if hasattr(msg, 'content')):
        return {
            "is_good_answer": True,
            "num_feedback_requests": num_feedback_requests + 1
        }

    try:
        judge_llm = base_llm.with_structured_output(JudgeOutput)
        system_prompt = SystemMessage(content=prompts.judge_prompt)
        response: JudgeOutput = judge_llm.invoke([system_prompt] + state["messages"])
        output = {
            "is_good_answer": response.is_good_answer,
            "num_feedback_requests": num_feedback_requests + 1,
        }
        if response.feedback:
            output["messages"] = [AIMessage(content=response.feedback)]
            print("Judge node output:", output["messages"][-1])
        return output
    except Exception as e:
        print(f"Error in judge_node: {e}")
        return {
            "is_good_answer": True,  # Assume good answer to avoid infinite loops
            "num_feedback_requests": num_feedback_requests + 1
        }
        
# Final answer router function
def final_answer_router(state: AgentState):
    """Router to determine the final answer to the user query."""
    #print("#" * 50)
    #print("Final answer router input:", state["messages"][-1])
    if state["is_good_answer"]:
        return "end"
    else:
        # Check if we've exceeded maximum planning cycles
        num_planning_cycles = state.get("num_planning_cycles", 0)
        max_planning_cycles = 3  # Maximum number of planning-agent-judge cycles
        
        if num_planning_cycles >= max_planning_cycles:
            #print("Final answer router - terminating due to max planning cycles")
            # Route to termination node instead of returning dict
            return "termination"
        else:
            #print("Final answer router - continuing to planning")
            return "planning"


def termination_node(state: AgentState):
    """Node to handle graceful termination when max cycles are reached."""
    #print("#" * 50)
    #print("Termination node input:", state["messages"][-1])
    termination_message = AIMessage(
        content="I've reached the maximum number of attempts to provide a satisfactory answer. "
               "While I may not have fully met your expectations, I've provided the best response "
               "possible with the available information and tools. Please let me know if you'd like "
               "me to try a different approach or if you have additional questions."
    )
    #print("Termination node output:", termination_message)
    return {
        "messages": [termination_message],
        "is_good_answer": True  # Force termination
    }

    
## Define the workflow(Graph)

workflow = StateGraph(AgentState)

## Add nodes to graph
workflow.add_node("decision_making", decision_making_node)
workflow.add_node("planning", planning_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.add_node("answer_compiler", answer_compiler)
workflow.add_node("judge", judge_node)
workflow.add_node("termination", termination_node)

# Set the entry point of the graph
workflow.set_entry_point("decision_making")

## Add edges between nodes
workflow.add_conditional_edges(
    "decision_making",
    router,
    {"planning": "planning", "end": END}
)
workflow.add_edge("planning", "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue":"tools", "end": "answer_compiler"}
)
workflow.add_edge("answer_compiler", "judge")
workflow.add_conditional_edges(
    "judge",
    final_answer_router,
    {"end": END, "planning": "planning", "termination": "termination"}
)
workflow.add_edge("termination", END)

# compile the graph
climadoc = workflow.compile(debug=True)
    
# Wrapper function to handle invocation properly
def run_climadoc_workflow(query: str):
    """
    Wrapper function to run the research workflow with proper error handling
    """
    try:
        # Validate input query
        if not query or not query.strip():
            return {
                "messages": [AIMessage(content="I apologize, but I didn't receive a valid query. Please provide a question or request.")]
            }
        
        initial_state = {
            "query": [HumanMessage(content=query.strip())],
            "messages": [HumanMessage(content=query.strip())]
        }
        
        
        # Use invoke with proper configuration
        result = climadoc.invoke(initial_state, config={"recursion_limit": 50})
        return result
        
    except Exception as e:
        print(f"Error in research workflow: {e}")
        import traceback
        traceback.print_exc()
        return {
            "messages": [AIMessage(content=f"I encountered an error while processing your request: {str(e)}. Please try again with a different question.")]
        }