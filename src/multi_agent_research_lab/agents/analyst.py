"""Analyst agent — turns research notes into structured insights."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a critical analyst. Given research notes and the original query, produce a "
    "structured analysis (300-500 words) with these sections:\n"
    "## Key Claims — bullet list of the most important factual claims.\n"
    "## Evidence Strength — rate each claim: Strong / Moderate / Weak.\n"
    "## Gaps & Contradictions — what is missing or uncertain.\n"
    "## Synthesis — 2-3 sentences connecting the dots for the audience."
)


class AnalystAgent(BaseAgent):
    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.errors.append("analyst: no research_notes available")
            return state

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            "Produce a structured analysis."
        )

        with trace_span("analyst.llm_call") as span:
            resp = self._llm.complete(_SYSTEM, user_prompt)
            span["attributes"]["tokens_out"] = resp.output_tokens

        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=resp.content,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
            )
        )
        state.add_trace_event("analyst.done", {"analysis_len": len(resp.content)})
        return state
