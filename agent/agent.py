"""Fetches GGUF metadata (revision, filename, size) from HuggingFace
and appends a new model entry to benchmark_config.yaml."""

import yaml
from pathlib import Path
from typing import TypedDict, Optional
from huggingface_hub import HfApi
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    model_id: str                        
    config_path: str                     
    hf_metadata: Optional[dict]          
    updated_config: Optional[dict]       
    error: Optional[str]                 


def fetch_metadata(state: AgentState) -> dict:
    """
    Node 1: Search HuggingFace for a GGUF repo matching this model,
    then find the Q4_K_M file and grab its revision + size.
    """
    print(f"\n[1/3] Fetching metadata for {state['model_id']} from HuggingFace...")

    api = HfApi()
    model_id = state["model_id"]
    model_name = model_id.split("/")[-1]  
    author = model_id.split("/")[0]

    try:
        results = api.list_models(
            search=f"{model_name} GGUF",
            author=author,
        )
        gguf_repos = [r.id for r in results]

        if not gguf_repos:
            return {"error": f"No GGUF repo found for {model_id} by author {author}"}

        quantized_repo = gguf_repos[0]
        print(f"    found GGUF repo: {quantized_repo}")

        repo_info = api.repo_info(repo_id=quantized_repo, repo_type="model")
        revision = repo_info.sha

        files = list(api.list_repo_files(repo_id=quantized_repo))

        
        # preference order: Q4_K_M is the sweet spot for quality vs size
        # fall back to other quantizations if that's not available
        PREFERRED = ["Q4_K_M", "Q4_0", "Q5_K_M", "Q8_0"]

        gguf_file = None
        quantization_used = None
        for quant in PREFERRED:
            for f in files:
                if quant in f and f.endswith(".gguf"):
                    gguf_file = f
                    quantization_used = quant
                    break
            if gguf_file:
                break

        if not gguf_file:
            return {"error": f"No GGUF file found in {quantized_repo}"}

        print(f"    quantization: {quantization_used}")

        file_info = api.get_paths_info(repo_id=quantized_repo, paths=[gguf_file])
        size_bytes = file_info[0].size

        print(f"    revision:  {revision}")
        print(f"    gguf file: {gguf_file}")
        print(f"    size:      {size_bytes:,} bytes")

        return {
            "hf_metadata": {
                "quantized_repo": quantized_repo,
                "revision": revision,
                "filename": gguf_file,
                "size_bytes": size_bytes,
                "quantization_used": quantization_used,
            }
        }

    except Exception as e:
        return {"error": str(e)}

def build_config_entry(state: AgentState) -> dict:
    """
    Node 2: Using what HuggingFace gave us, construct the new model entry
    and slot it into the existing config dict.

    We never rewrite the whole config from scratch — we read what's already
    there and add one new block under 'models:'. That way all the existing
    datasets, prompts, adapters etc. stay untouched.
    """
    print(f"\n[2/3] Building config entry...")

    #if prev node hits error, skip this one
    if state.get("error"):
        return {}

    model_id = state["model_id"]
    meta = state["hf_metadata"]

    short_name = model_id.split("/")[-1].lower().replace(".", "").replace("_", "-")

    local_path = f".models/{short_name.replace('-', '_')}/{meta['filename']}"


    #which adapter to use
    adapter = "qwen3_llama_cpp" if "qwen3" in model_id.lower() else "llama_cpp"

    new_entry = {
        "adapter": adapter,
        "display_name": model_id.split("/")[-1],
        "model_id": model_id,
        "quantized_repo": meta["quantized_repo"],
        "revision": meta["revision"],
        "filename": meta["filename"],
        "path": local_path,
        "expected_size_bytes": meta["size_bytes"],
        "quantization": f"{meta.get('quantization_used', 'Q4_K_M')} GGUF",
    }

    print(f"    key:      {short_name}")
    print(f"    adapter:  {adapter}")
    print(f"    path:     {local_path}")

    config_path = Path(state["config_path"])
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["models"][short_name] = new_entry

    return {"updated_config": config}


def write_config(state: AgentState) -> dict:
    """
    Node 3: Write the updated config dict back to benchmark_config.yaml.
    This is the last node — after this the graph ends.
    """
    print(f"\n[3/3] Writing updated config...")

    if state.get("error"):
        print(f"\n    ERROR: {state['error']}")
        print("    Config was NOT modified.")
        return {}

    config_path = Path(state["config_path"])
    with open(config_path, "w") as f:
        yaml.dump(state["updated_config"], f, default_flow_style=False, sort_keys=False)

    print(f"    Wrote to {config_path}")
    print(f"\nDone! Added {state['model_id']} to benchmark_config.yaml")
    return {}


"""wire the nodes together"""

def build_graph():
    graph = StateGraph(AgentState)

    # Register each node with a name and the function it runs
    graph.add_node("fetch_metadata", fetch_metadata)
    graph.add_node("build_config_entry", build_config_entry)
    graph.add_node("write_config", write_config)

    # Define the edges — what runs after what
    graph.set_entry_point("fetch_metadata")
    graph.add_edge("fetch_metadata", "build_config_entry")
    graph.add_edge("build_config_entry", "write_config")
    graph.add_edge("write_config", END)

    return graph.compile()



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 agent.py <model_id> <path_to_config>")
        print("Example: python3 agent.py Qwen/Qwen3-2.5B ../benchmark/benchmark_config.yaml")
        sys.exit(1)

    app = build_graph()

    result = app.invoke({
        "model_id": sys.argv[1],
        "config_path": sys.argv[2],
        "hf_metadata": None,
        "updated_config": None,
        "error": None,
    })

    if result.get("error"):
        sys.exit(1)
