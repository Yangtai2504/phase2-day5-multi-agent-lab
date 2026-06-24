"""LangGraph multi-agent workflow.

Graph topology:
  [supervisor] --researcher--> [researcher] -+
               --analyst-----> [analyst]    -+-> back to [supervisor]
               --writer------> [writer]     -+
               --done--------> END
"""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


def _make_node(agent_run):
    """Wrap an agent.run call so it accepts and returns a plain dict."""
    def node(state_dict: dict[str, Any]) -> dict[str, Any]:
        state = ResearchState(**state_dict)
        updated = agent_run(state)
        return updated.model_dump()
    return node


def _route(state_dict: dict[str, Any]) -> str:
    """Read the last route token written by the supervisor."""
    history: list[str] = state_dict.get("route_history", [])
    last = history[-1] if history else ROUTE_RESEARCHER
    if last == ROUTE_DONE:
        return END
    return last


class MultiAgentWorkflow:
    """Builds and runs the LangGraph multi-agent graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()

    def build(self) -> Any:
        """Create and compile the LangGraph graph."""
        # Use dict state internally so LangGraph can merge cleanly
        graph: StateGraph = StateGraph(dict)

        graph.add_node("supervisor", _make_node(self._supervisor.run))
        graph.add_node("researcher", _make_node(self._researcher.run))
        graph.add_node("analyst", _make_node(self._analyst.run))
        graph.add_node("writer", _make_node(self._writer.run))

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            _route,
            {
                ROUTE_RESEARCHER: "researcher",
                ROUTE_ANALYST: "analyst",
                ROUTE_WRITER: "writer",
                END: END,
            },
        )

        # Workers always return to supervisor after completing
        for worker in ("researcher", "analyst", "writer"):
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile the graph, invoke it, and return the final ResearchState."""
        compiled = self.build()
        result_dict = compiled.invoke(state.model_dump())
        return ResearchState(**result_dict)
