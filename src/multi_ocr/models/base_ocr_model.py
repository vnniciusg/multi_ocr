from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from PIL.Image import Image

from ..config import OCRModelConfig


@dataclass
class OCRResult:
    text: str
    reasoning: Optional[str] = field(default=None)
    raw_output: Optional[str] = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        parts = [f"text_length={len(self.text)}"]
        if self.reasoning:
            parts.append(f"reasoning_tokens={len(self.reasoning.split())}")

        return f"OCRResult({', '.join(parts)})"


class BaseOCRModel(ABC):
    def __init__(self, config: OCRModelConfig) -> None:
        self._config = config
        self._model = None
        self._processor = None
        self._tokenizer = None

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def process_image(
        self, image: Image, prompt: Optional[str] = None, **kwargs
    ) -> OCRResult: ...

    def _get_device(self) -> str:
        if self._config.device:
            return self._config.device

        return "cuda" if torch.cuda.is_available() else "cpu"
