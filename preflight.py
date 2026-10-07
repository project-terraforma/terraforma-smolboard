import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys

import torch

import benchmark
import checkpoint
import runtime


class Report:
    def __init__(self):
        self.warnings = []
        self.failures = []

    def ok(self, msg):
        print(f"  ok    {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def fail(self, msg):
        self.failures.append(msg)
        print(f"  FAIL  {msg}")


def check_python(rep):
    desc = f"Python {platform.python_version()} on {platform.platform()}"
    if (3, 10) <= sys.version_info[:2] <= (3, 13):
        rep.ok(desc)
    else:
        rep.warn(f"{desc}: requirements.txt is tested on Python 3.10-3.13 (pandas==2.2.3 has no newer wheels)")


_PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)\s*(?:;([^#]*))?")


def pinned_versions():
    from packaging.markers import Marker
    pins = {}
    with open(os.path.join(benchmark.PROJECT_DIR, "requirements.txt"), encoding="utf-8") as f:
        for line in f:
            m = _PIN.match(line.strip())
            if m and (not m.group(3) or Marker(m.group(3).strip()).evaluate()):
                pins[m.group(1)] = m.group(2)
    return pins


def check_versions(rep):
    pins = pinned_versions()
    wrong = []
    for name, want in pins.items():
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            wrong.append(f"{name} missing (want {want})")
            continue
        if have.split("+")[0] != want:
            wrong.append(f"{name} {have} (want {want})")
    if wrong:
        rep.warn("installed versions differ from requirements.txt, so generations may not match the pinned "
                 "benchmark: " + "; ".join(wrong))
    else:
        rep.ok(f"all {len(pins)} pinned library versions match requirements.txt")


def _unused_gpu():
    if sys.platform == "darwin":
        if platform.machine() == "arm64" and not torch.backends.mps.is_available():
            return (f"this Apple Silicon Mac's GPU (MPS) is unavailable to torch (built with MPS: "
                    f"{torch.backends.mps.is_built()}, macOS {platform.mac_ver()[0]})")
        return None
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            gpus = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            gpus = ""
        if gpus:
            if torch.version.cuda:
                why = (f"torch (built for CUDA {torch.version.cuda}) cannot initialise it -- is the NVIDIA "
                       f"driver too old for this build?")
            else:
                why = (f"torch {torch.__version__} is a CPU-only build (on Windows the PyPI wheel always is). "
                       f"Install the CUDA build of the same version, e.g. `pip install torch=={torch.__version__.split('+')[0]} "
                       f"--index-url https://download.pytorch.org/whl/cu126` (see requirements.txt)")
            return f"an NVIDIA GPU is present ({gpus.splitlines()[0]}) but {why}"
    if (shutil.which("rocm-smi") or os.path.exists("/dev/kfd")) and not getattr(torch.version, "hip", None):
        return ("an AMD GPU seems present but this torch build has no ROCm support; install the ROCm build "
                "from https://pytorch.org/get-started/locally/")
    return None


def check_accelerator(rep, profile):
    build = (f"CUDA {torch.version.cuda}" if torch.version.cuda
             else f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None)
             else "CPU/MPS build")
    if profile.backend != "cpu":
        rep.ok(f"torch {torch.__version__} ({build}) on {profile.backend}: {profile.device_name}")
        return
    unused = _unused_gpu()
    if unused:
        rep.warn(f"running on the CPU (days for the full benchmark) although {unused}")
    else:
        rep.ok(f"torch {torch.__version__} ({build}): no usable GPU, running on the CPU (slow)")


def _nf4_relative_error(profile):
    import bitsandbytes as bnb
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(512, 256, generator=gen) * 0.05
    layer = bnb.nn.Linear4bit(256, 512, bias=False, compute_dtype=profile.dtype,
                              compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(weight.to(profile.dtype), requires_grad=False,
                                     compress_statistics=True, quant_type="nf4")
    layer = layer.to(profile.backend)
    reference_weight = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).float()
    worst = 0.0
    for rows in (1, 8):
        x = torch.randn(rows, 256, generator=gen).to(device=profile.backend, dtype=profile.dtype)
        with torch.no_grad():
            y = layer(x).float()
        if not torch.isfinite(y).all():
            return float("nan")
        reference = x.float() @ reference_weight.t()
        worst = max(worst, float((y - reference).norm() / reference.norm()))
    return worst


def check_nf4(rep, profile):
    if profile.quantization != "nf4":
        rep.ok(f"unquantized {profile.dtype_name} weights: bitsandbytes is not used")
        return
    import bitsandbytes as bnb
    what = f"bitsandbytes {bnb.__version__} NF4 with {profile.dtype_name} compute on {profile.backend}"
    try:
        err = _nf4_relative_error(profile)
    except Exception as e:
        rep.fail(f"{what} does not run: {type(e).__name__}: {e}")
        return
    if not err < 0.05:
        rep.fail(f"{what} returns wrong results (relative error {err:.3g} against its own dequantized weights)")
    else:
        rep.ok(f"{what} works ({runtime.bnb_kernel_path(profile.backend)} kernels)")


def check_dataset(rep):
    try:
        path = benchmark.ensure_dataset()
    except Exception as e:
        rep.fail(f"golden dataset: {e}")
        return None
    rep.ok(f"dataset {os.path.relpath(path, benchmark.PROJECT_DIR)} "
           f"(sha256 OK, terraforma-smolboard@{benchmark.TERRAFORMA_COMMIT[:12]})")
    return benchmark.load_rows()


def check_prompts(rep, rows):
    import dspy
    import pyarrow
    dataset = benchmark.build_dataset(rows, True, benchmark.N_PER_PAIR_TYPE, benchmark.SAMPLE_SEED)
    got = benchmark.prompt_fingerprints(dataset)
    changed = [c for c in benchmark.CONDITION_NAMES if got[c] != benchmark.PROMPT_SHA256[c]]
    if changed:
        rep.fail(f"the {'/'.join(changed)} messages built on this machine differ from the frozen benchmark "
                 f"(installed: dspy {dspy.__version__}, pyarrow {pyarrow.__version__}; pinned with dspy 3.3.1). "
                 f"Install requirements.txt. Only for an intended change, update PROMPT_SHA256 in benchmark.py.")
    else:
        rep.ok(f"json/text/dspy messages for all {len(dataset)} pairs are identical to the frozen benchmark")


def check_hf_auth(rep):
    from huggingface_hub import HfApi, constants, get_token
    if constants.HF_HUB_OFFLINE:
        rep.ok("Hugging Face: offline mode (HF_HUB_OFFLINE) -- cached files only, login not checked")
        return
    if get_token() is None:
        rep.warn("Hugging Face: not logged in -- the gated models (gemma-3, llama-3.2) will fail to load; "
                 "run `hf auth login` or set HF_TOKEN")
        return
    try:
        rep.ok(f"Hugging Face: logged in as {HfApi().whoami()['name']}")
    except Exception as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 401:
            rep.warn("Hugging Face: the token (HF_TOKEN or `hf auth login`) was rejected; log in again")
        else:
            rep.warn(f"Hugging Face: could not reach huggingface.co to check the login ({type(e).__name__}: {e})")


def _cached_weight_bytes():
    from huggingface_hub import scan_cache_dir
    try:
        repos = scan_cache_dir().repos
    except Exception:
        return {}
    return {
        (repo.repo_id, rev.commit_hash): sum(f.size_on_disk for f in rev.files)
        for repo in repos if repo.repo_type == "model"
        for rev in repo.revisions
        if any(f.file_name.endswith(".safetensors") for f in rev.files)
    }


def check_models(rep, active_models):
    from huggingface_hub import auth_check, constants
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    offline = constants.HF_HUB_OFFLINE
    cached = _cached_weight_bytes()
    missing, problems = {}, []
    print("  models (pinned revisions):")
    for short_name, repo_id in active_models.items():
        size = cached.get((repo_id, benchmark.MODEL_REVISIONS[short_name]))
        if not size:
            missing[short_name] = repo_id
        if offline:
            status = "cached" if size else "NOT CACHED"
        else:
            try:
                auth_check(repo_id)
                status = "access OK"
            except GatedRepoError:
                status = "NO ACCESS (gated)"
            except RepositoryNotFoundError:
                status = "NO ACCESS"
            except Exception as e:
                status = f"UNCHECKED ({type(e).__name__})"
        if status not in ("cached", "access OK"):
            problems.append(short_name)
        print(f"        {short_name:26s} {repo_id:40s} {status:18s} "
              + (f"cached {size / 1e9:.1f} GB" if size else "not downloaded yet"))
    if problems and offline:
        rep.warn(f"offline mode, but not in the local cache: {', '.join(problems)}")
    elif problems:
        rep.warn(f"no confirmed access to {', '.join(problems)}. Gated repos need their license accepted on "
                 f"huggingface.co by the logged-in account")
    else:
        rep.ok(f"all {len(active_models)} selected model(s) " + ("cached" if offline else "accessible"))
    return missing


def check_disk(rep, missing, delete_weights=False):
    from huggingface_hub import HfApi, constants
    if not missing:
        rep.ok("disk: every selected model is already cached")
        return
    api = HfApi()
    sizes = []
    for short_name, repo_id in missing.items():
        try:
            info = api.model_info(repo_id, revision=benchmark.MODEL_REVISIONS[short_name], files_metadata=True)
        except Exception:
            continue
        sizes.append(sum(s.size or 0 for s in info.siblings
                         if "/" not in s.rfilename and s.rfilename.startswith("model")
                         and s.rfilename.endswith(".safetensors")))
    need = max(sizes, default=0) if delete_weights else sum(sizes)
    probe = constants.HF_HUB_CACHE
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    msg = (f"disk: {free / 1e9:.0f} GB free for the HF cache ({constants.HF_HUB_CACHE}); "
           f"{len(missing)} model(s) still to download, ~{need / 1e9:.0f} GB"
           + (" at a time (--delete-weights)" if delete_weights else ""))
    if need > free:
        rep.warn(msg + ". Free some space, point HF_HOME at a bigger disk, or use --delete-weights")
    else:
        rep.ok(msg)


def check_prompt_version(rep, checkpoint_path):
    current = benchmark.PROMPT_VERSION
    versions = {r.get("prompt_version", "v1") for r in checkpoint.read_runtime_records(checkpoint_path)}
    try:
        results = checkpoint.load_results(checkpoint_path)
    except ValueError as e:
        rep.fail(str(e))
        return
    answered = int((~results["status"].isin(checkpoint.RETRYABLE_STATUSES)).sum()) if len(results) else 0
    if answered and not versions:
        versions = {"v1"}
    other = sorted(versions - {current})
    if other:
        rep.fail(f"{checkpoint_path} holds rows from prompt version {', '.join(other)} (this code is {current}); "
                 f"they are a different experiment. Use another --checkpoint (the default is named by version).")
    else:
        rep.ok(f"checkpoint holds only prompt {current} rows ({answered} answered so far)")


def check_checkpoint(rep, checkpoint_path):
    try:
        lock = checkpoint.CheckpointLock(checkpoint_path)
    except RuntimeError as e:
        rep.warn(str(e))
        return
    except OSError as e:
        rep.fail(f"cannot write next to the checkpoint: {type(e).__name__}: {e}")
        return
    lock.release()
    rep.ok("checkpoint directory is writable and no other run is using this checkpoint")


def run(profile, active_models, checkpoint_path, full, delete_weights=False):
    from huggingface_hub import constants
    rep = Report()
    print("preflight:")
    check_python(rep)
    check_versions(rep)
    check_accelerator(rep, profile)
    check_nf4(rep, profile)
    rows = check_dataset(rep)
    if rows is not None:
        check_prompts(rep, rows)
    check_prompt_version(rep, checkpoint_path)
    check_hf_auth(rep)
    if full:
        check_checkpoint(rep, checkpoint_path)
        missing = check_models(rep, active_models)
        if not constants.HF_HUB_OFFLINE:
            check_disk(rep, missing, delete_weights)
    return rep
