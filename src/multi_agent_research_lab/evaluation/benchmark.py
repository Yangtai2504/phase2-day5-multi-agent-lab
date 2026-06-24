"""Benchmark single-agent vs multi-agent runs."""

import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def _total_cost(state: ResearchState) -> float:
    """Sum cost_usd across all agent results."""
    return sum(
        r.metadata.get("cost_usd") or 0.0
        for r in state.agent_results
    )


def _citation_coverage(state: ResearchState) -> float:
    """Fraction of paragraphs in final_answer that contain a [Source N] citation."""
    if not state.final_answer:
        return 0.0
    paragraphs = [p.strip() for p in state.final_answer.split("\n") if p.strip()]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[Source\s*\d+\]", p))
    return round(cited / len(paragraphs), 2)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner(query)`, measure wall-clock time, cost, and citation coverage."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    cost = _total_cost(state)
    citation_cov = _citation_coverage(state)

    notes = (
        f"agents={len(state.agent_results)}, "
        f"sources={len(state.sources)}, "
        f"citations={citation_cov:.0%}, "
        f"errors={len(state.errors)}"
    )

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=round(cost, 6) if cost else None,
        notes=notes,
    )
    return state, metrics
