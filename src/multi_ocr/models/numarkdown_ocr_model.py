import re
from typing import ClassVar, Optional

import torch
from PIL.Image import Image

from .base_ocr_model import BaseOCRModel, OCRResult


class NumarkdownOCRModel(BaseOCRModel):
    THINKING_REGEX_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"<think>(?P<think>.*?)</think>", re.DOTALL
    )
    ANSWER_REGEX_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"<answer>(?P<answer>.*?)</answer>", re.DOTALL
    )

    def load_model(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = self._get_device()

        self._processor = AutoProcessor.from_pretrained(
            self._config.ocr_model_id,
            trust_remote_code=self._config.trust_remote_code,
            min_pixels=self._config.min_pixels,
            max_pixels=self._config.max_pixels,
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._config.ocr_model_id,
            torch_dtype=self._config.torch_dtype,
            attn_implementation=self.config.attn_implementation,
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=self.config.trust_remote_code,
        )

    def process_image(
        self, image: Image, prompt: Optional[str] = None, **kwargs
    ) -> OCRResult:
        _text_prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        _inputs = self.processor(
            text=[_text_prompt], images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **_inputs,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            )

        _raw_output = self.processor.decode(output_ids[0], skip_special_tokens=True)
        _content, _reasoning = self._split_awnser_and_reasoning(raw_output=_raw_output)

        return OCRResult(text=_content, reasoning=_reasoning, raw_output=_raw_output)

    def _split_answer_and_reasoning(
        self, *, raw_output: str
    ) -> tuple[Optional[str], Optional[str]]:
        _reasoning_match = self.THINKING_REGEX_PATTERN.search(raw_output)
        _answer_match = self.ANSWER_REGEX_PATTERN.search(raw_output)

        return _reasoning_match.group("think").strip(), _answer_match.group(
            "answer"
        ).strip()
