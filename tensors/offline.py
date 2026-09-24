"""Fail-closed local model discovery for network-isolated clusters."""

from pathlib import Path
import os
from urllib.parse import urlparse


MODEL_ENV_VARS = ("TENSORCACHE_MODEL_PATH", "TRELLIS_MODEL_PATH")
DEFAULT_MODEL_ID = "microsoft/TRELLIS-image-large"


def configure_offline_environment() -> None:
    """Force common ML libraries into offline mode before they are imported."""
    defaults = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "SPCONV_ALGO": "native",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _huggingface_hub_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_local_model_path(explicit_path=None, model_id=DEFAULT_MODEL_ID) -> str:
    """Resolve a local snapshot without contacting Hugging Face."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    for variable in MODEL_ENV_VARS:
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]).expanduser())

    model_cache = _huggingface_hub_root() / f"models--{model_id.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    if snapshots.is_dir():
        candidates.extend(
            sorted(
                (path for path in snapshots.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved / "pipeline.json").is_file():
            return str(resolved)

    searched = "\n  - ".join(str(path) for path in candidates) or "(no candidates)"
    raise FileNotFoundError(
        "TRELLIS local model snapshot with pipeline.json was not found. "
        "Network fallback is disabled.\n"
        "Pass --model_path /local/snapshot or set TENSORCACHE_MODEL_PATH.\n"
        f"Searched:\n  - {searched}"
    )


def patch_torch_hub_offline() -> None:
    """Allow only local torch.hub repositories and already-cached checkpoints."""
    import torch
    import torch.hub

    if getattr(torch.hub, "_tensorcache_offline", False):
        return

    original_load = torch.hub.load
    original_load_state = torch.hub.load_state_dict_from_url

    def local_load(repo_or_dir, model, *args, source="github", **kwargs):
        direct = Path(str(repo_or_dir)).expanduser()
        if source == "local" or direct.is_dir():
            return original_load(str(direct), model, *args, source="local", **kwargs)

        prefix = str(repo_or_dir).replace("/", "_").replace(":", "_") + "_"
        hub_root = Path(torch.hub.get_dir())
        matches = sorted(
            (path for path in hub_root.glob(prefix + "*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            raise FileNotFoundError(
                f"torch.hub repository {repo_or_dir!r} is not cached under {hub_root}; "
                "network fallback is disabled"
            )
        return original_load(str(matches[0]), model, *args, source="local", **kwargs)

    def cached_state_dict(url, *args, **kwargs):
        filename = kwargs.get("file_name") or Path(urlparse(url).path).name
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint {filename!r} is not cached at {checkpoint}; "
                "network fallback is disabled"
            )
        return original_load_state(url, *args, **kwargs)

    torch.hub.load = local_load
    torch.hub.load_state_dict_from_url = cached_state_dict
    torch.hub._tensorcache_offline = True
