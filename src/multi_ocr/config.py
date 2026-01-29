from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Optional

import torch

type OCRModelID = Literal["numind/NuMarkdown-8B-Thinking"]


class OCRModelType(StrEnum):
    NUMARKDOWN = "numarkdown"


@dataclass
class OCRModelConfig:
    ocr_model_id: OCRModelID
    ocr_model_type: OCRModelType
    device: Optional[str] = field(default=None)
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = field(default="flash_attention_2")
    trust_remote_code: bool = True
    temperature: float = 0.7
    max_new_tokens: int = 5000

    # MODEL SPECIFIC PARAMETERS
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    custom_params: dict[str, Any] = field(default_factory=dict)


OCR_PRESET_CONFIGS: dict[OCRModelType, OCRModelConfig] = {
    OCRModelType.NUMARKDOWN: OCRModelConfig(
        ocr_model_id="numind/NuMarkdown-8B-Thinking",
        ocr_model_type=OCRModelType.NUMARKDOWN,
        min_pixels=100 * 28 * 28,
        max_pixels=5000 * 28 * 28,
    )
}
    
