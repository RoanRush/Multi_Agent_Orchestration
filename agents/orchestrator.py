import operator
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agents.researcher import get_researcher_agent
from agents.analyst import get_analyst_agent
from agents.writer import get_writer_agent
from tools.search_tool import search_tool
from tools.calculator_tool import calculator_tool
from tools.summarizer_tool import summarizer_tool


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    current_agent: str
    task: str
    research_output: str
    analysis_output: str
    final_output: str
    iteration: int



def researcher_node(state: AgentState) -> AgentState:
    """Performs information gathering and research."""
    agent = get_researcher_agent()
    task = state["task"]

    result = agent.invoke({
        "input": f"Research the following topic thoroughly:\n{task}",
        "chat_history": state["messages"],
    })

    research_output = result.get("output", "")

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=f"[Researcher]: {research_output}")],
        "research_output": research_output,
        "current_agent": "analyst",
        "iteration": state.get("iteration", 0) + 1,
    }


def analyst_node(state: AgentState) -> AgentState:
    """Analyzes and synthesizes research output."""
    agent = get_analyst_agent()
    research = state.get("research_output", "")

    result = agent.invoke({
        "input": (
            f"Analyze the following research and extract key insights:\n\n"
            f"Research:\n{research}\n\n"
            f"Original Task:\n{state['task']}"
        ),
        "chat_history": state["messages"],
    })

    analysis_output = result.get("output", "")

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=f"[Analyst]: {analysis_output}")],
        "analysis_output": analysis_output,
        "current_agent": "writer",
        "iteration": state.get("iteration", 0) + 1,
    }


def writer_node(state: AgentState) -> AgentState:
    """Generates the final polished response."""
    agent = get_writer_agent()
    analysis = state.get("analysis_output", "")
    research = state.get("research_output", "")

    result = agent.invoke({
        "input": (
            f"Write a clear, well-structured response based on:\n\n"
            f"Research:\n{research}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Task:\n{state['task']}"
        ),
        "chat_history": state["messages"],
    })

    final_output = result.get("output", "")

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=f"[Writer]: {final_output}")],
        "final_output": final_output,
        "current_agent": "end",
        "iteration": state.get("iteration", 0) + 1,
    }


def build_orchestrator() -> StateGraph:
    tools = [search_tool, calculator_tool, summarizer_tool]
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("researcher")

    graph.add_conditional_edges(
        "researcher",
        lambda s: "analyst" if s.get("research_output") else "tools",
    )
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)
    graph.add_edge("tools", "researcher")

    return graph.compile()


orchestrator = build_orchestrator()


def run_orchestrator(task: str, chat_history: list[BaseMessage] | None = None) -> dict:
    """
    Entry point: run the full multi-agent pipeline for a given task.

    Args:
        task: Natural language task description.
        chat_history: Optional prior conversation context.

    Returns:
        Dict with final_output and full message history.
    """
    initial_state: AgentState = {
        "messages": chat_history or [HumanMessage(content=task)],
        "current_agent": "researcher",
        "task": task,
        "research_output": "",
        "analysis_output": "",
        "final_output": "",
        "iteration": 0,
    }

    result = orchestrator.invoke(initial_state)
    return {
        "final_output": result["final_output"],
        "messages": result["messages"],
        "iterations": result["iteration"],
    }
