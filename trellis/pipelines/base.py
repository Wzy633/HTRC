from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model from a complete local snapshot.
        """
        import json
        from pathlib import Path

        snapshot = Path(path).expanduser().resolve()
        config_file = snapshot / "pipeline.json"
        if not config_file.is_file():
            raise FileNotFoundError(
                f"local TRELLIS pipeline config is missing: {config_file}; "
                "network fallback is disabled")

        with config_file.open('r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            model_prefix = Path(v).expanduser()
            if not model_prefix.is_absolute():
                model_prefix = snapshot / model_prefix
            _models[k] = models.from_pretrained(str(model_prefix))

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
