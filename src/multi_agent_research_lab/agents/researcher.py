"""Researcher agent — searches sources and distils research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM = (
    "You are a research specialist. Given a query and a list of source snippets, "
    "write concise, factual research notes (400-600 words). "
    "Cite each claim with [Source N] where N is the 1-based index of the source. "
    "Focus on accuracy; omit filler."
)


class ResearcherAgent(BaseAgent):
    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient(self._llm)

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher.search") as span:
            sources = self._search.search(
                state.request.query,
                max_results=state.request.max_sources,
            )
            span["attributes"]["source_count"] = len(sources)

        state.sources = sources

        source_block = "\n".join(
            f"[{i + 1}] {s.title}: {s.snippet}" for i, s in enumerate(sources)
        )
        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"Sources:\n{source_block}\n\n"
            "Write research notes."
        )

        with trace_span("researcher.llm_call") as span:
            resp = self._llm.complete(_SYSTEM, user_prompt)
            span["attributes"]["tokens_out"] = resp.output_tokens

        state.research_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
            )
        )
        state.add_trace_event("researcher.done", {"notes_len": len(resp.content)})
        return state
