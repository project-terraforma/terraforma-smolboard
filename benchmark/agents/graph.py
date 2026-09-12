"""LangGraph-style orchestration for benchmark scheduling."""

from __future__ import annotations

from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency for local CLI usage
    StateGraph = None
    START = "__start__"
    END = "__end__"


class BenchmarkState(TypedDict, total=False):
    models: list[str]
    model_index: int
    results: list[dict[str, Any]]


def _next_step(state: dict[str, Any]) -> str:
    models = state.get("models") or []
    model_index = int(state.get("model_index", 0) or 0)
    return "run_model" if model_index < len(models) else "write_report"


def build_graph() -> Any | None:
    """Create a minimal LangGraph graph when the dependency is installed."""
    if StateGraph is None:
        return None

    graph = StateGraph(BenchmarkState)
    graph.add_node("run_model", lambda state: state)
    graph.add_node("write_report", lambda state: state)
    graph.add_edge(START, "run_model")
    graph.add_conditional_edges(
        "run_model",
        _next_step,
        {
            "run_model": "run_model",
            "write_report": "write_report",
        },
    )
    graph.add_edge("write_report", END)
    return graph.compile()
