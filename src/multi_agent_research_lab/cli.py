"""Command-line entrypoint for the lab."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Single-agent baseline: one LLM call does everything."""
    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()

    system = (
        "You are a research assistant. Answer the query thoroughly (500-700 words) "
        "with citations in [Source N] format. Provide a References section."
    )
    resp = llm.complete(system, f"Query: {query}")
    state.final_answer = resp.content

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full Supervisor → Researcher → Analyst → Writer pipeline."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    if result.errors:
        console.print(Panel.fit("\n".join(result.errors), title="Errors", style="red"))

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Result"))
    console.print(f"\n[dim]Route: {' -> '.join(result.route_history)}[/dim]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output markdown file")] = Path("reports/benchmark_report.md"),
) -> None:
    """Benchmark single-agent vs multi-agent and save a markdown report."""
    _init()
    console.print("[bold]Running single-agent baseline…[/bold]")

    def run_baseline(q: str) -> ResearchState:
        req = ResearchQuery(query=q)
        st = ResearchState(request=req)
        llm = LLMClient()
        system = (
            "You are a research assistant. Answer the query thoroughly (500-700 words) "
            "with citations in [Source N] format."
        )
        resp = llm.complete(system, f"Query: {q}")
        st.final_answer = resp.content
        return st

    _, baseline_metrics = run_benchmark("single-agent", query, run_baseline)
    console.print(f"  Baseline: {baseline_metrics.latency_seconds:.1f}s, cost={baseline_metrics.estimated_cost_usd}")

    console.print("[bold]Running multi-agent pipeline…[/bold]")

    def run_multi(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(st)

    _, multi_metrics = run_benchmark("multi-agent", query, run_multi)
    console.print(f"  Multi-agent: {multi_metrics.latency_seconds:.1f}s, cost={multi_metrics.estimated_cost_usd}")

    report = render_markdown_report([baseline_metrics, multi_metrics])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    console.print(f"\n[green]Report saved to {out}[/green]")
    console.print(report)


if __name__ == "__main__":
    app()
