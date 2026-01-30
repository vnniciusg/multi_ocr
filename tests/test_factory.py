from multi_ocr import MultiOCRFactory
from multi_ocr._config import OCRModelType
from multi_ocr._ocr_models import (
    GotOCRModel,
    LightOnOCRModel,
)


class TestMultiOCRFactory:
    def test_returns_got_ocr_model(self):
        cls = MultiOCRFactory.get_ocr_model(ocr_model_type=OCRModelType.GOT_OCR)
        assert cls is GotOCRModel

    def test_returns_lighton_ocr_model(self):
        cls = MultiOCRFactory.get_ocr_model(ocr_model_type=OCRModelType.LIGHTON_OCR)
        assert cls is LightOnOCRModel

    def test_returns_none_for_unknown_type(self):
        result = MultiOCRFactory.get_ocr_model(ocr_model_type="nonexistent")
        assert result is None

    def test_registry_contains_all_model_types(self):
        registry = MultiOCRFactory._MODEL_REGISTRY
        assert set(registry.keys()) == set(OCRModelType)
