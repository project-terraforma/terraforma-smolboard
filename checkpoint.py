import io
import json
import os
import shutil

import pandas as pd

try:
    import fcntl
except ImportError:
    fcntl = None

CHECKPOINT_COLUMNS = [
    "model", "prompt", "example_id", "pair_type", "hard_neg_source",
    "true_label", "raw_output", "prediction", "status", "correct",
]
KEY_COLUMNS = ["model", "prompt", "example_id", "status"]

RETRYABLE_STATUSES = {"model_load_error", "generation_error"}


def _read_checkpoint_repairing_torn_tail(path, write_back=True):
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    n_torn_lines = 0
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        n_torn_lines = 1

    def write_repaired(head, n_lines):
        if write_back:
            tmp = path + ".repair"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(head)
            os.replace(tmp, path)
            print(f"  repaired {path}: dropped a torn half-written tail ({n_lines} line(s)); that row will be re-run")

    if not text.strip():
        return pd.DataFrame(), n_torn_lines
    try:
        df = pd.read_csv(io.StringIO(text))
        if n_torn_lines:
            write_repaired(text, n_torn_lines)
        return df, n_torn_lines
    except pd.errors.ParserError as e:
        original_error = e
    lines = text.splitlines(keepends=True)
    for cut in range(1, min(len(lines), 500)):
        head = "".join(lines[:-cut])
        try:
            df = pd.read_csv(io.StringIO(head))
        except pd.errors.ParserError:
            continue
        write_repaired(head, cut + n_torn_lines)
        return df, cut + n_torn_lines
    raise original_error


def load_checkpoint(path):
    if os.path.exists(path) and os.path.getsize(path) == 0:
        os.remove(path)
    if os.path.exists(path):
        df, _ = _read_checkpoint_repairing_torn_tail(path)
        if list(df.columns) != CHECKPOINT_COLUMNS:
            if len(df) == 0:
                os.remove(path)
                print(f"  {path} held only a torn header; starting it fresh")
                return pd.DataFrame(columns=CHECKPOINT_COLUMNS), set()
            raise ValueError(
                f"{path} is not a KR3 checkpoint: columns {list(df.columns)}, expected {CHECKPOINT_COLUMNS}"
            )
        n_raw = len(df)
        df = df.dropna(subset=[c for c in KEY_COLUMNS if c in df.columns]).reset_index(drop=True)
        if len(df) != n_raw:
            print(f"  dropped {n_raw - len(df)} torn/incomplete row(s) from {path}; they'll be re-run")
        if "example_id" in df.columns:
            df["example_id"] = df["example_id"].astype(int)
        attempted = {
            (r.model, r.prompt, r.example_id)
            for r in df.itertuples()
            if r.status not in RETRYABLE_STATUSES
        }
        print(f"resuming from {path}: {len(df)} rows on disk, {len(attempted)} generations already attempted")
        return df, attempted
    print(f"no checkpoint at {path} yet -- starting fresh")
    return pd.DataFrame(columns=CHECKPOINT_COLUMNS), set()


def append_rows(path, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    text = pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS).to_csv(header=new_file, index=False, lineterminator="\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


class CheckpointLock:

    """Exclusive lock on <checkpoint>.lock for the life of the process. The OS releases it
    when the process exits, however it exits, so a stale lock file is harmless."""
    def __init__(self, checkpoint_path):
        self.path = checkpoint_path + ".lock"
        self._f = None
        if fcntl is None:
            from filelock import FileLock, Timeout
            self._f = FileLock(self.path)
            try:
                self._f.acquire(timeout=0)
            except Timeout:
                raise RuntimeError(
                    f"another run is already writing {checkpoint_path}. Two writers would duplicate "
                    f"work and corrupt the file; stop the other run or use a different --checkpoint."
                ) from None
            return
        self._f = open(self.path, "a+")
        try:
            fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._f.seek(0)
            holder = self._f.read().strip() or "?"
            self._f.close()
            raise RuntimeError(
                f"another run (pid {holder}) is already writing {checkpoint_path}. Two writers would "
                f"duplicate work and corrupt the file; stop the other run or use a different --checkpoint."
            ) from None
        self._f.seek(0)
        self._f.truncate()
        self._f.write(str(os.getpid()))
        self._f.flush()

    def release(self):
        if fcntl is None:
            self._f.release()
        else:
            self._f.close()


def load_results(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        final_df, _ = _read_checkpoint_repairing_torn_tail(path, write_back=False)
        if list(final_df.columns) != CHECKPOINT_COLUMNS:
            if final_df.empty:
                return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
            raise ValueError(f"{path} is not a KR3 checkpoint: columns {list(final_df.columns)}")
        final_df = final_df.dropna(subset=KEY_COLUMNS)
        final_df = final_df.drop_duplicates(subset=["model", "prompt", "example_id"], keep="last").reset_index(drop=True)
        return final_df
    return pd.DataFrame(columns=CHECKPOINT_COLUMNS)


def runtime_log_path(checkpoint_path):
    return os.path.splitext(checkpoint_path)[0] + ".runtime.jsonl"


def append_runtime_record(checkpoint_path, record):
    with open(runtime_log_path(checkpoint_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_runtime_records(checkpoint_path):
    path = runtime_log_path(checkpoint_path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HubMirror:

    REVISION = "checkpoint"

    def __init__(self, checkpoint_path, repo_id=None, every_seconds=120):
        from huggingface_hub import CommitScheduler, HfApi
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        try:
            api = HfApi()
            user = api.whoami()["name"]
        except Exception as e:
            raise RuntimeError(
                "--hub-sync needs a valid Hugging Face login (HF_TOKEN env var or `hf auth login`)."
            ) from e

        self.repo_id = repo_id or f"{user}/kr3-benchmark-checkpoints"
        folder = os.path.dirname(os.path.abspath(checkpoint_path))
        names = [os.path.basename(checkpoint_path), os.path.basename(runtime_log_path(checkpoint_path))]

        api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.create_branch(repo_id=self.repo_id, repo_type="dataset", branch=self.REVISION, exist_ok=True)

        for name in names:
            local = os.path.join(folder, name)
            if os.path.exists(local):
                print(f"local {name} present -- not overwriting it from the Hub")
                continue
            try:
                cached = api.hf_hub_download(repo_id=self.repo_id, repo_type="dataset",
                                             filename=name, revision=self.REVISION)
                shutil.copy(cached, local)
                print(f"restored {name} from hf.co/datasets/{self.repo_id}")
            except (EntryNotFoundError, RepositoryNotFoundError):
                print(f"no existing {name} in hf.co/datasets/{self.repo_id} yet")

        self._scheduler = CommitScheduler(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=True,
            folder_path=folder,
            path_in_repo="",
            revision=self.REVISION,
            allow_patterns=names,
            every=every_seconds / 60,
        )
        print(f"checkpoint mirrored to hf.co/datasets/{self.repo_id} (branch={self.REVISION}, "
              f"every {every_seconds}s, plus after every model and on interruption)")

    def sync_now(self, reason):
        try:
            self._scheduler.trigger().result(timeout=60)
        except Exception as e:
            print(f"  (checkpoint sync to the Hub after {reason} failed, will retry on the "
                  f"next scheduled push: {type(e).__name__}: {e})")
