"""Writer agent — synthesises a polished final answer with citations."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a technical writer. Given a query, research notes, and an analysis, "
    "write a clear, well-structured response (500-700 words) for the stated audience. "
    "Requirements:\n"
    "- Start with a one-paragraph executive summary.\n"
    "- Use markdown headers and bullet points where appropriate.\n"
    "- Cite sources inline as [Source N] wherever you make factual claims.\n"
    "- End with a 'References' section listing the sources used.\n"
    "Do NOT fabricate sources beyond what was provided."
)


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes or not state.analysis_notes:
            state.errors.append("writer: missing research_notes or analysis_notes")
            return state

        source_list = "\n".join(
            f"[{i + 1}] {s.title} — {s.url or 'no URL'}" for i, s in enumerate(state.sources)
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{state.analysis_notes}\n\n"
            f"Available sources:\n{source_list}\n\n"
            "Write the final response."
        )

        with trace_span("writer.llm_call") as span:
            resp = self._llm.complete(_SYSTEM, user_prompt, max_tokens=2048)
            span["attributes"]["tokens_out"] = resp.output_tokens

        state.final_answer = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd},
            )
        )
        state.add_trace_event("writer.done", {"answer_len": len(resp.content)})
        return state
