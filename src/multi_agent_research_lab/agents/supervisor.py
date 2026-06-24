"""Supervisor / router — decides which worker runs next."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

# Routing tokens used by the workflow's conditional edge
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Stateless router: inspects state and emits the next route token."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        settings = get_settings()

        # Hard stop on max iterations to prevent infinite loops
        if state.iteration >= settings.max_iterations:
            state.record_route(ROUTE_DONE)
            state.errors.append(f"supervisor: max_iterations ({settings.max_iterations}) reached")
            return state

        # If a previous step recorded an error that blocked progress, stop
        if state.errors and state.iteration > 0:
            state.record_route(ROUTE_DONE)
            return state

        # Sequential pipeline: research → analyse → write → done
        if state.research_notes is None:
            route = ROUTE_RESEARCHER
        elif state.analysis_notes is None:
            route = ROUTE_ANALYST
        elif state.final_answer is None:
            route = ROUTE_WRITER
        else:
            route = ROUTE_DONE

        state.record_route(route)
        state.add_trace_event("supervisor.route", {"route": route, "iteration": state.iteration})
        return state
