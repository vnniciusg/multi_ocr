import pytest
from PIL import Image

from multi_ocr._config import OCRModelConfig, OCRModelType


@pytest.fixture()
def sample_config():
    return OCRModelConfig(
        ocr_model_id="stepfun-ai/GOT-OCR-2.0-hf",
        ocr_model_type=OCRModelType.GOT_OCR,
    )


@pytest.fixture()
def tmp_image():
    return Image.new("RGB", (64, 64), color="white")
