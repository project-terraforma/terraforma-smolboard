"""Runtime configuration for the benchmark agent.

Fill in the values in this file or set the equivalent environment variables before
running the agent workflow. This keeps API keys and repo credentials out of the
workflow code itself.
"""

from __future__ import annotations

import os
from typing import Any


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    return default


AGENT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": _env("AGENT_LLM_PROVIDER", "openai"),
        "model": _env("AGENT_LLM_MODEL", "gpt-4.1-mini"),
        "api_key": _env("OPENAI_API_KEY"),
        "base_url": _env("OPENAI_BASE_URL"),
    },
    "git": {
        "repo_url": _env("BENCHMARK_REPO_URL"),
        "branch": _env("BENCHMARK_REPO_BRANCH", "main"),
        "username": _env("BENCHMARK_GIT_USERNAME"),
        "token": _env("BENCHMARK_GIT_TOKEN"),
    },
    "agent": {
        "workspace_root": _env("BENCHMARK_WORKSPACE_ROOT"),
        "data_dir": _env("BENCHMARK_DATA_DIR", "website/data"),
        "commit_message": _env("BENCHMARK_COMMIT_MESSAGE", "Update benchmark leaderboard data"),
    },
}


def get_agent_config() -> dict[str, Any]:
    return AGENT_CONFIG


if __name__ == "__main__":
    print(AGENT_CONFIG)
