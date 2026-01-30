from typing import ClassVar

from ._config import OCR_PRESET_CONFIGS, OCRModelConfig, OCRModelID, OCRModelType
from ._ocr_models import (
    BaseOCRModel,
    GotOCRModel,
    LightOnOCRModel,
)


class MultiOCRFactory:
    _MODEL_REGISTRY: ClassVar[dict[OCRModelType, type[BaseOCRModel]]] = {
        OCRModelType.GOT_OCR: GotOCRModel,
        OCRModelType.LIGHTON_OCR: LightOnOCRModel,
    }

    @classmethod
    def get_ocr_model(cls, *, ocr_model_type: OCRModelType) -> type[BaseOCRModel]:
        return cls._MODEL_REGISTRY.get(ocr_model_type)


__all__ = [
    "OCR_PRESET_CONFIGS",
    "BaseOCRModel",
    "GotOCRModel",
    "LightOnOCRModel",
    "MultiOCRFactory",
    "OCRModelConfig",
    "OCRModelID",
    "OCRModelType",
]
