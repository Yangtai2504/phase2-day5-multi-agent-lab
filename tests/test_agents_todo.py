"""Tests for agent routing logic (supervisor implemented)."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE, ROUTE_RESEARCHER
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def test_supervisor_routes_researcher_first() -> None:
    state = _state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_RESEARCHER


def test_supervisor_routes_done_when_complete() -> None:
    state = _state()
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_DONE


def test_supervisor_stops_at_max_iterations() -> None:
    state = _state()
    state.iteration = 100  # force over max
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] == ROUTE_DONE
    assert any("max_iterations" in e for e in result.errors)
