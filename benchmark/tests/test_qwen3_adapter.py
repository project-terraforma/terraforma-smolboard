from smolboard_benchmark.adapters import create_adapter
from smolboard_benchmark.adapters.qwen3 import Qwen3LlamaCppAdapter


def test_qwen3_adapter_uses_non_thinking_template_setting():
    adapter = create_adapter(
        "qwen3-4b",
        {"adapter": "qwen3_llama_cpp"},
        {"parallel_slots": 1},
        {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8, "seed": 42, "stop": ["\n"]},
        allow_download=False,
    )
    assert isinstance(adapter, Qwen3LlamaCppAdapter)
    assert adapter._request_payload("classify this")["chat_template_kwargs"] == {"enable_thinking": False}
