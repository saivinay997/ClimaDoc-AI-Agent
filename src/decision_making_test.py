import pytest
from workflow import decision_making_node, AgentState

from langchain_core.messages import HumanMessage


# ---------- Helper Assertions ----------

def assert_weather(response):
    assert response["action"] == "weather"


def assert_rag(response):
    assert response["action"] == "rag"


def assert_direct(response):
    assert response["action"] == "direct"


# ---------- Weather Tool Tests ----------

@pytest.mark.parametrize("query", [
    "What's the weather in Bengaluru?",
    "Will it rain today?",
    "Temperature in Delhi right now",
    "Weather forecast for next week",
    "Is it windy outside?",
    "Do I need an umbrella today?"
])
def test_weather_queries(query):
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    
    assert_weather(response)


# ---------- RAG Tool Tests ----------

@pytest.mark.parametrize("query", [
    "According to the policy, can we work during rain?",
    "What does the safety document say?",
    "Summarize the uploaded PDF",
    "Does the manual mention heatwave rules?",
    "What are the guidelines mentioned in the document?"
])
def test_rag_queries(query):
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    
    assert_rag(response)


# ---------- Direct Answer Tests ----------

@pytest.mark.parametrize("query", [
    "Hi",
    "Hello ClimaDoc",
    "What can you do?",
    "Thank you",
    "Tell me a joke",
    "Who are you?"
])
def test_direct_queries(query):
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    
    assert_direct(response)


# ---------- Ambiguous Queries ----------

@pytest.mark.parametrize("query", [
    "Is it safe today?",
    "Should we cancel today?",
    "Is it allowed to work outside?"
])
def test_ambiguous_queries(query):
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    
    # Enforce deterministic routing
    assert response["action"] in {"weather", "rag"}
    assert response["messages"] is None


# ---------- Mixed Intent Queries ----------

@pytest.mark.parametrize("query", [
    "Based on today's weather, does policy allow outdoor work?",
    "Can we work today as per company rules?",
])
def test_mixed_intent_queries(query):
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    
    # RAG should dominate in mixed cases
    assert response["action"] == "rag"
    assert response["messages"] is None


# ---------- Output Schema Validation ----------

def test_response_schema():
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    
    assert set(response.keys()) == {"action", "messages"}
    assert response["action"] in {"weather", "rag", "direct"}

    if response["action"] == "direct":
        assert isinstance(response["messages"], str)
    else:
        assert response["messages"] is None


# ---------- Robustness Tests ----------

@pytest.mark.parametrize("query", [
    "",
    "   ",
    "???",
    "asdfghjkl",
])
def test_invalid_or_garbage_input(query):
    state = {"messages": [HumanMessage(content=query)]}
    state = AgentState(**state)
    response = decision_making_node(state)
    

    # Should not crash
    assert response["action"] in {"weather", "rag", "direct"}
