from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from multi_ocr.__main__ import build_config, main
from multi_ocr._config import OCR_PRESET_CONFIGS, OCRModelType
from multi_ocr._ocr_models import OCRResult


class TestBuildConfig:
    def test_returns_preset_config_when_no_overrides(self):
        config = build_config(OCRModelType.GOT_OCR, None, None, None)
        assert config is OCR_PRESET_CONFIGS[OCRModelType.GOT_OCR]

    def test_applies_device_override(self):
        config = build_config(OCRModelType.GOT_OCR, "cuda", None, None)
        assert config.device == "cuda"
        assert config.ocr_model_id == "stepfun-ai/GOT-OCR-2.0-hf"

    def test_applies_temperature_override(self):
        config = build_config(OCRModelType.GOT_OCR, None, 0.3, None)
        assert config.temperature == 0.3

    def test_applies_max_new_tokens_override(self):
        config = build_config(OCRModelType.GOT_OCR, None, None, 2000)
        assert config.max_new_tokens == 2000

    def test_applies_multiple_overrides(self):
        config = build_config(OCRModelType.GOT_OCR, "cpu", 0.5, 1000)
        assert config.device == "cpu"
        assert config.temperature == 0.5
        assert config.max_new_tokens == 1000

    def test_raises_click_exception_for_missing_preset(self):
        fake_type = MagicMock()
        fake_type.value = "fake-model"
        with pytest.raises(click.ClickException, match="No preset config found"):
            build_config(fake_type, None, None, None)


class TestCLIMain:
    @patch("multi_ocr.__main__.MultiOCRFactory")
    @patch("multi_ocr.__main__.Image")
    def test_valid_invocation(self, mock_pil, mock_factory, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_model_instance = MagicMock()
        mock_model_instance.process_image.return_value = OCRResult(text="OCR output text")
        mock_model_cls = MagicMock(return_value=mock_model_instance)
        mock_factory.get_ocr_model.return_value = mock_model_cls

        runner = CliRunner()
        result = runner.invoke(main, ["got-ocr", str(img_file)])

        assert result.exit_code == 0
        assert "OCR output text" in result.output
        mock_model_instance.load_model.assert_called_once()
        mock_model_instance.process_image.assert_called_once()

    @patch("multi_ocr.__main__.MultiOCRFactory")
    @patch("multi_ocr.__main__.Image")
    def test_output_file_writing(self, mock_pil, mock_factory, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        out_file = tmp_path / "output.txt"

        mock_model_instance = MagicMock()
        mock_model_instance.process_image.return_value = OCRResult(text="saved text")
        mock_model_cls = MagicMock(return_value=mock_model_instance)
        mock_factory.get_ocr_model.return_value = mock_model_cls

        runner = CliRunner()
        result = runner.invoke(main, ["got-ocr", str(img_file), "-o", str(out_file)])

        assert result.exit_code == 0
        assert out_file.read_text() == "saved text"
        assert "Result saved to" in result.output

    @patch("multi_ocr.__main__.MultiOCRFactory")
    @patch("multi_ocr.__main__.Image")
    def test_show_reasoning_flag(self, mock_pil, mock_factory, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_model_instance = MagicMock()
        mock_model_instance.process_image.return_value = OCRResult(
            text="result", reasoning="model reasoning here"
        )
        mock_model_cls = MagicMock(return_value=mock_model_instance)
        mock_factory.get_ocr_model.return_value = mock_model_cls

        runner = CliRunner()
        result = runner.invoke(main, ["got-ocr", str(img_file), "--show-reasoning"])

        assert result.exit_code == 0
        assert "model reasoning here" in result.output
        assert "--- Reasoning ---" in result.output
