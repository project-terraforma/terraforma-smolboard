"""Git publication helpers for the benchmark data export."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agents.config import get_agent_config


def git_repo_url() -> str:
    cfg = get_agent_config()
    url = (cfg.get("git") or {}).get("repo_url")
    if not url:
        raise ValueError("Set BENCHMARK_REPO_URL or fill the git.repo_url config")
    return url


def git_push_data(workspace_root: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Stage and push the generated JSON data files to the repository.

    This is intentionally simple and does not commit any benchmark artifacts that are
    not inside the configured data directory. Add your credentials via environment
    variables or by editing the config file in this package.
    """
    root = Path(workspace_root)
    data_path = root / data_dir
    cfg = get_agent_config()
    git_cfg = cfg.get("git") or {}
    token = git_cfg.get("token")
    username = git_cfg.get("username")
    repo_url = git_cfg.get("repo_url")

    if not repo_url:
        raise ValueError("Missing repo URL: set BENCHMARK_REPO_URL or edit benchmark/agents/config.py")

    if token and username and "https://" in repo_url:
        auth_url = repo_url.replace("https://", f"https://{username}:{token}@")
    else:
        auth_url = repo_url

    subprocess.run(["git", "-C", str(root), "add", str(data_path)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-m", cfg.get("agent", {}).get("commit_message", "Update benchmark data")],
        check=False,
    )
    subprocess.run(["git", "-C", str(root), "remote", "set-url", "origin", auth_url], check=False)
    subprocess.run(["git", "-C", str(root), "push", "origin", git_cfg.get("branch", "main")], check=False)

    return {"repo_url": repo_url, "data_dir": str(data_path), "pushed": True}
