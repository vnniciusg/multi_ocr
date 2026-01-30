from unittest.mock import MagicMock, patch

import torch

from multi_ocr._config import OCRModelConfig, OCRModelType
from multi_ocr._ocr_models import (
    GotOCRModel,
    LightOnOCRModel,
    OCRResult,
)


class TestOCRResult:
    def test_str_returns_text(self):
        result = OCRResult(text="hello world")
        assert str(result) == "hello world"

    def test_repr_text_only(self):
        result = OCRResult(text="hello world")
        assert result.__repr__() == "OCRResult(text_length=11)"

    def test_repr_with_reasoning(self):
        result = OCRResult(text="hello", reasoning="step one step two")
        r = repr(result)
        assert "text_length=5" in r
        assert "reasoning_tokens=4" in r

    def test_default_optional_fields(self):
        result = OCRResult(text="x")
        assert result.reasoning is None
        assert result.raw_output is None
        assert result.metadata == {}

    def test_metadata_field(self):
        result = OCRResult(text="x", metadata={"model": "got"})
        assert result.metadata == {"model": "got"}


def _make_model(cls, **config_overrides):
    defaults = dict(
        ocr_model_id="stepfun-ai/GOT-OCR-2.0-hf",
        ocr_model_type=OCRModelType.GOT_OCR,
    )
    defaults.update(config_overrides)
    config = OCRModelConfig(**defaults)
    return cls(config)


class TestBaseOCRModelGetDevice:
    def test_returns_config_device_when_set(self):
        model = _make_model(GotOCRModel, device="cpu")
        assert model._get_device() == "cpu"

    @patch("multi_ocr._ocr_models.torch")
    def test_falls_back_to_cuda(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        model = _make_model(GotOCRModel)
        assert model._get_device() == "cuda"

    @patch("multi_ocr._ocr_models.torch")
    def test_falls_back_to_mps(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        model = _make_model(GotOCRModel)
        assert model._get_device() == "mps"

    @patch("multi_ocr._ocr_models.torch")
    def test_falls_back_to_cpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        model = _make_model(GotOCRModel)
        assert model._get_device() == "cpu"


class TestBaseOCRModelKwargs:
    def test_model_kwargs_cuda_device_map(self):
        model = _make_model(GotOCRModel, device="cuda")
        kwargs = model._model_kwargs()
        assert kwargs["device_map"] == "auto"
        assert kwargs["torch_dtype"] == torch.bfloat16
        assert kwargs["trust_remote_code"] is True
        assert kwargs["attn_implementation"] == "sdpa"

    def test_model_kwargs_cpu_device_map(self):
        model = _make_model(GotOCRModel, device="cpu")
        kwargs = model._model_kwargs()
        assert kwargs["device_map"] == "cpu"

    def test_processor_kwargs_default(self):
        model = _make_model(GotOCRModel)
        kwargs = model._processor_kwargs()
        assert kwargs == {"trust_remote_code": True}


class TestBaseOCRModelLoadModel:
    @patch("multi_ocr._ocr_models.BaseOCRModel._load_component")
    def test_load_model_calls_components(self, mock_load):
        mock_load.return_value = MagicMock()
        model = _make_model(GotOCRModel)
        model.load_model()

        calls = mock_load.call_args_list
        class_names = [c[0][0] for c in calls]
        assert "AutoProcessor" in class_names
        assert "AutoModelForImageTextToText" in class_names
        assert model._processor is not None
        assert model._model is not None


class TestGotOCRModel:
    def test_class_variables(self):
        assert GotOCRModel.model_class == "AutoModelForImageTextToText"
        assert GotOCRModel.processor_class == "AutoProcessor"

    def test_process_image(self, tmp_image):
        model = _make_model(GotOCRModel, device="cpu")

        mock_processor = MagicMock()
        mock_model = MagicMock()

        input_ids = torch.tensor([[1, 2, 3]])
        mock_processor.return_value = MagicMock(**{"to.return_value": {"input_ids": input_ids}})
        mock_processor.decode.return_value = "extracted text"

        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        model._processor = mock_processor
        model._model = mock_model

        result = model.process_image(tmp_image)
        assert result.text == "extracted text"
        assert result.raw_output == "extracted text"
        mock_processor.assert_called_once()
        mock_model.generate.assert_called_once()


class TestLightOnOCRModel:
    def test_processor_kwargs_empty(self):
        config = OCRModelConfig(
            ocr_model_id="lightonai/LightOnOCR-2-1B",
            ocr_model_type=OCRModelType.LIGHTON_OCR,
        )
        model = LightOnOCRModel(config)
        assert model._processor_kwargs() == {}

    def test_model_kwargs_float32_on_mps(self):
        config = OCRModelConfig(
            ocr_model_id="lightonai/LightOnOCR-2-1B",
            ocr_model_type=OCRModelType.LIGHTON_OCR,
            device="mps",
        )
        model = LightOnOCRModel(config)
        kwargs = model._model_kwargs()
        assert kwargs["torch_dtype"] == torch.float32
        assert kwargs["device_map"] == "mps"

    def test_model_kwargs_cuda(self):
        config = OCRModelConfig(
            ocr_model_id="lightonai/LightOnOCR-2-1B",
            ocr_model_type=OCRModelType.LIGHTON_OCR,
            device="cuda",
        )
        model = LightOnOCRModel(config)
        kwargs = model._model_kwargs()
        assert kwargs["torch_dtype"] == torch.bfloat16
        assert kwargs["device_map"] == "auto"

    def test_process_image_message_structure(self, tmp_image):
        config = OCRModelConfig(
            ocr_model_id="lightonai/LightOnOCR-2-1B",
            ocr_model_type=OCRModelType.LIGHTON_OCR,
            device="cpu",
        )
        model = LightOnOCRModel(config)

        mock_processor = MagicMock()
        mock_model = MagicMock()

        input_ids = torch.tensor([[1, 2, 3]])
        mock_processor.apply_chat_template.return_value = {
            "input_ids": input_ids,
        }
        mock_processor.decode.return_value = "lighton result"

        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        model._processor = mock_processor
        model._model = mock_model

        result = model.process_image(tmp_image)
        assert result.text == "lighton result"
        assert result.raw_output == "lighton result"

        call_args = mock_processor.apply_chat_template.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "image"
