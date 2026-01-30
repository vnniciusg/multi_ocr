from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import torch

type OCRModelID = Literal[
    "stepfun-ai/GOT-OCR-2.0-hf",
    "lightonai/LightOnOCR-2-1B",
]


class OCRModelType(StrEnum):
    GOT_OCR = "got-ocr"
    LIGHTON_OCR = "lighton-ocr"


@dataclass
class OCRModelConfig:
    ocr_model_id: OCRModelID
    ocr_model_type: OCRModelType
    device: str | None = field(default=None)
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = field(default="sdpa")
    trust_remote_code: bool = True
    temperature: float = 0.7
    max_new_tokens: int = 5000
    custom_params: dict[str, Any] = field(default_factory=dict)


OCR_PRESET_CONFIGS: dict[OCRModelType, OCRModelConfig] = {
    OCRModelType.GOT_OCR: OCRModelConfig(
        ocr_model_id="stepfun-ai/GOT-OCR-2.0-hf",
        ocr_model_type=OCRModelType.GOT_OCR,
        attn_implementation="eager",
    ),
    OCRModelType.LIGHTON_OCR: OCRModelConfig(
        ocr_model_id="lightonai/LightOnOCR-2-1B",
        ocr_model_type=OCRModelType.LIGHTON_OCR,
    ),
}
