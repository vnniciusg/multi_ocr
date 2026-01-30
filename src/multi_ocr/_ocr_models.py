from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from PIL.Image import Image

from ._config import OCRModelConfig


@dataclass
class OCRResult:
    text: str
    reasoning: str | None = field(default=None)
    raw_output: str | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        parts = [f"text_length={len(self.text)}"]
        if self.reasoning:
            parts.append(f"reasoning_tokens={len(self.reasoning.split())}")

        return f"OCRResult({', '.join(parts)})"


class BaseOCRModel(ABC):
    model_class: ClassVar[str]
    processor_class: ClassVar[str | None] = "AutoProcessor"
    tokenizer_class: ClassVar[str | None] = None

    def __init__(self, config: OCRModelConfig) -> None:
        self._config = config
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _load_component(self, class_name: str, **kwargs: Any):
        import transformers  # noqa: PLC0415

        cls = getattr(transformers, class_name)
        return cls.from_pretrained(self._config.ocr_model_id, **kwargs)

    def _processor_kwargs(self) -> dict[str, Any]:
        return {"trust_remote_code": self._config.trust_remote_code}

    def _model_kwargs(self) -> dict[str, Any]:
        device = self._get_device()
        return {
            "torch_dtype": self._config.torch_dtype,
            "attn_implementation": self._config.attn_implementation,
            "device_map": "auto" if device == "cuda" else device,
            "trust_remote_code": self._config.trust_remote_code,
        }

    def load_model(self) -> None:
        if self.processor_class:
            self._processor = self._load_component(self.processor_class, **self._processor_kwargs())

        if self.tokenizer_class:
            self._tokenizer = self._load_component(
                self.tokenizer_class,
                trust_remote_code=self._config.trust_remote_code,
            )

        self._model = self._load_component(self.model_class, **self._model_kwargs())

    @abstractmethod
    def process_image(
        self, image: Image, prompt: str | None = None, **kwargs: Any
    ) -> OCRResult: ...

    def _get_device(self) -> str:
        if self._config.device:
            return self._config.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class GotOCRModel(BaseOCRModel):
    model_class = "AutoModelForImageTextToText"

    def process_image(self, image: Image, prompt: str | None = None, **kwargs: Any) -> OCRResult:
        inputs = self._processor(image, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=kwargs.get("max_new_tokens", self._config.max_new_tokens),
            )

        raw_output = self._processor.decode(
            output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return OCRResult(text=raw_output, raw_output=raw_output)


class LightOnOCRModel(BaseOCRModel):
    model_class = "LightOnOcrForConditionalGeneration"
    processor_class = "LightOnOcrProcessor"

    def _processor_kwargs(self) -> dict[str, Any]:
        return {}

    def _model_kwargs(self) -> dict[str, Any]:
        device = self._get_device()
        dtype = torch.float32 if device == "mps" else self._config.torch_dtype
        return {
            "torch_dtype": dtype,
            "device_map": "auto" if device == "cuda" else device,
        }

    def process_image(self, image: Image, prompt: str | None = None, **kwargs: Any) -> OCRResult:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = self._model.device
        dtype = self._config.torch_dtype
        if device.type == "mps":
            dtype = torch.float32

        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self._config.max_new_tokens),
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        raw_output = self._processor.decode(generated_ids, skip_special_tokens=True)
        return OCRResult(text=raw_output, raw_output=raw_output)
