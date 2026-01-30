import torch

from multi_ocr._config import OCR_PRESET_CONFIGS, OCRModelConfig, OCRModelType


class TestOCRModelType:
    def test_got_ocr_value(self):
        assert OCRModelType.GOT_OCR == "got-ocr"

    def test_lighton_ocr_value(self):
        assert OCRModelType.LIGHTON_OCR == "lighton-ocr"

    def test_all_types_exist(self):
        assert len(OCRModelType) == 2

    def test_is_str_enum(self):
        assert isinstance(OCRModelType.GOT_OCR, str)


class TestOCRModelConfig:
    def test_default_values(self):
        config = OCRModelConfig(
            ocr_model_id="stepfun-ai/GOT-OCR-2.0-hf",
            ocr_model_type=OCRModelType.GOT_OCR,
        )
        assert config.device is None
        assert config.torch_dtype == torch.bfloat16
        assert config.attn_implementation == "sdpa"
        assert config.trust_remote_code is True
        assert config.temperature == 0.7
        assert config.max_new_tokens == 5000
        assert config.custom_params == {}


class TestOCRPresetConfigs:
    def test_contains_all_model_types(self):
        assert set(OCR_PRESET_CONFIGS.keys()) == set(OCRModelType)

    def test_got_preset_model_id(self):
        config = OCR_PRESET_CONFIGS[OCRModelType.GOT_OCR]
        assert config.ocr_model_id == "stepfun-ai/GOT-OCR-2.0-hf"
        assert config.ocr_model_type == OCRModelType.GOT_OCR

    def test_got_preset_uses_eager_attention(self):
        config = OCR_PRESET_CONFIGS[OCRModelType.GOT_OCR]
        assert config.attn_implementation == "eager"

    def test_lighton_preset_model_id(self):
        config = OCR_PRESET_CONFIGS[OCRModelType.LIGHTON_OCR]
        assert config.ocr_model_id == "lightonai/LightOnOCR-2-1B"
        assert config.ocr_model_type == OCRModelType.LIGHTON_OCR
