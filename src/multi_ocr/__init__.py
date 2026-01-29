from typing import ClassVar

from .config import OCRModelID, OCRModelConfig, OCRModelType, OCR_PRESET_CONFIGS
from .models.base_ocr_model import BaseOCRModel
from .models.numarkdown_ocr_model import NumarkdownOCRModel


class MultiOCRFactory:
    _MODEL_REGISTRY: ClassVar[dict[OCRModelType, BaseOCRModel]] = {
        OCRModelType.NUMARKDOWN: NumarkdownOCRModel
    }

    @classmethod
    def get_ocr_model(cls, *, ocr_model_type: OCRModelType) -> BaseOCRModel:
        return cls._MODEL_REGISTRY.get(ocr_model_type)


__all__ = [
    "OCRModelID",
    "OCRModelConfig",
    "OCRModelType",
    "OCR_PRESET_CONFIGS",
    "NumarkdownOCRModel",
    "MultiOCRFactory",
]
