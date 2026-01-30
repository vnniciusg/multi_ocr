# Multi OCR

A Python library that provides a unified interface for multiple OCR models.

## Supported Models

| Model         | ID                          | Type flag      |
| ------------- | --------------------------- | -------------- |
| GOT-OCR 2.0   | `stepfun-ai/GOT-OCR-2.0-hf` | `got-ocr`      |
| DeepSeek-OCR  | `deepseek-ai/DeepSeek-OCR`  | `deepseek-ocr` |
| LightOn-OCR 2 | `lightonai/LightOnOCR-2-1B` | `lighton-ocr`  |

## Requirements

- Python >= 3.13
- PyTorch, Transformers, Pillow

## Installation

```bash
uv sync
```

## CLI Usage

```bash
multi-ocr <model> <image> [OPTIONS]
```

Example:

```bash
multi-ocr got-ocr photo.png
```

Options:

- `--device` — Force device (`cuda`, `cpu`, `mps`). Auto-detected by default.
- `--max-new-tokens` — Maximum tokens to generate.
- `--show-reasoning` — Display model reasoning if available.
- `-o, --output` — Save result to a file.

## Library Usage

```python
from multi_ocr import MultiOCRFactory, OCR_PRESET_CONFIGS, OCRModelType
from PIL import Image

config = OCR_PRESET_CONFIGS[OCRModelType.GOT_OCR]
model_cls = MultiOCRFactory.get_ocr_model(config.ocr_model_type)
model = model_cls(config)
model.load_model()

result = model.process_image(Image.open("photo.png"))
print(result.text)
```

## License

[MIT](LICENSE)
