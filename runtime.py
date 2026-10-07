import gc
import os
import platform
import subprocess
from dataclasses import dataclass, field

import torch
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BACKENDS = ("cuda", "mps", "xpu", "cpu")
QUANTIZATIONS = ("nf4", "none")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class RuntimeProfile:
    backend: str
    quantization: str
    dtype: torch.dtype
    device_name: str
    notes: list = field(default_factory=list)

    @property
    def dtype_name(self):
        return str(self.dtype).replace("torch.", "")

    @property
    def label(self):
        return f"{self.backend}-{self.quantization}-{self.dtype_name}"

    @property
    def precision(self):
        """What the checkpoint file is keyed on: rows produced with different weights
        (4-bit NF4 vs. plain bf16/fp16) are different experiments and must not mix."""
        return self.quantization if self.quantization == "nf4" else self.dtype_name

    def device_map(self):
        return "auto" if self.backend == "cuda" else {"": self.backend}

    def describe(self):
        import accelerate, bitsandbytes, dspy, transformers
        return {
            "label": self.label,
            "backend": self.backend,
            "device_name": self.device_name,
            "quantization": self.quantization,
            "dtype": self.dtype_name,
            "bnb_kernels": bnb_kernel_path(self.backend) if self.quantization == "nf4" else None,
            "notes": self.notes,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "dspy": dspy.__version__,
        }


def detect_backend():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def _device_name(backend):
    if backend == "cuda":
        return torch.cuda.get_device_name(0)
    if backend == "xpu":
        return torch.xpu.get_device_name(0)
    if backend == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            chip = f"Apple {platform.machine()}"
        return f"{chip} (macOS {platform.mac_ver()[0]})"
    return platform.processor() or platform.machine()


def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def resolve_profile(backend="auto", quantization="auto", dtype="auto"):
    backend = detect_backend() if backend == "auto" else backend
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--backend cuda requested but torch.cuda.is_available() is False")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--backend mps requested but torch.backends.mps.is_available() is False")
    if backend == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("--backend xpu requested but torch.xpu.is_available() is False")

    quantization = "nf4" if quantization == "auto" else quantization
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"unknown quantization {quantization!r}; expected one of {QUANTIZATIONS}")

    if dtype != "auto":
        torch_dtype = DTYPES[dtype]
    elif backend == "cuda":
        torch_dtype = pick_compute_dtype()
    else:
        torch_dtype = torch.bfloat16

    notes = []
    if backend == "cuda" and getattr(torch.version, "hip", None):
        notes.append("AMD ROCm build of torch (exposed as cuda); bitsandbytes ROCm kernels.")
    if quantization == "nf4" and backend != "cuda":
        notes.append(
            f"Same NF4 + double-quant config as CUDA, run by bitsandbytes' {backend} kernels "
            f"({bnb_kernel_path(backend)}): near-identical, not guaranteed bit-identical, outputs."
        )
    if backend == "mps" and quantization == "nf4" and not bnb_kernel_path(backend).startswith("hub-metal"):
        notes.append(
            "bitsandbytes is using its pure-PyTorch MPS fallback: ~2x slower, and its raw text can differ "
            "slightly from sessions that used the Metal kernels (verdicts agreed 100% in testing). The "
            "Metal kernels need macOS 26+, `pip install kernels`, and network access (not HF_HUB_OFFLINE)."
        )
    if torch_dtype == torch.float16:
        notes.append(
            "float16 compute, as on the original Colab T4. Gemma-3 overflows in fp16 (NaN logits; that is why "
            "all 3,000 Colab T4 gemma rows are empty). Those generations are now recorded as retryable "
            "generation_errors and the model is skipped after 10; run it separately with "
            "`--models gemma-3-4b-it --dtype float32` (or bfloat16 where the GPU supports it)."
        )
    if quantization == "none":
        notes.append(f"NOT the reference config: unquantized {str(torch_dtype).replace('torch.', '')} weights.")

    return RuntimeProfile(backend, quantization, torch_dtype, _device_name(backend), notes)


def bnb_kernel_path(backend):
    """Which bitsandbytes 4-bit implementation this backend gets -- recorded in the
    runtime log because it is the one thing that differs from the CUDA reference."""
    if backend == "mps":
        try:
            from bitsandbytes.backends.mps import ops as mps_ops
            kernel = mps_ops._get_kernel()
        except Exception:
            return "unknown"
        if kernel is None:
            return "pytorch-fallback"
        parts = os.path.normpath(getattr(kernel, "__file__", "")).split(os.sep)
        snap = parts[parts.index("snapshots") + 1][:12] if "snapshots" in parts else "?"
        return f"hub-metal@{snap}"
    return {"cuda": "cuda", "xpu": "xpu", "cpu": "cpu"}[backend]


def load_model(repo_id: str, profile: RuntimeProfile, revision: str = None):
    kwargs = {}
    if profile.quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=profile.dtype,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        **kwargs,
        dtype=profile.dtype,
        device_map=profile.device_map(),
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _accelerators():
    if torch.cuda.is_available():
        yield torch.cuda
    if torch.backends.mps.is_available():
        yield torch.mps
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        yield torch.xpu


def empty_cache():
    for acc in _accelerators():
        try:
            acc.empty_cache()
        except Exception:
            pass


def free_model():
    gc.collect()
    empty_cache()
    for acc in _accelerators():
        try:
            acc.synchronize()
        except Exception:
            pass
    gc.collect()


def delete_cached_weights(repo_id):
    try:
        cache = scan_cache_dir()
        for cached in cache.repos:
            if cached.repo_type == "model" and cached.repo_id.lower() == repo_id.lower():
                strategy = cache.delete_revisions(*[rev.commit_hash for rev in cached.revisions])
                strategy.execute()
                print(f"  freed ~{strategy.expected_freed_size / 1e9:.1f} GB of disk: removed cached weights for {repo_id}")
    except Exception as e:
        print(f"  (could not remove cached weights for {repo_id}: {type(e).__name__}: {e})")
