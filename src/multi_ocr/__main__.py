from dataclasses import replace
from pathlib import Path

import click
from PIL import Image

from . import MultiOCRFactory, OCRModelType, OCR_PRESET_CONFIGS


def build_config(
    model_type: OCRModelType,
    device: str | None,
    temperature: float | None,
    max_new_tokens: int | None,
):
    config = OCR_PRESET_CONFIGS.get(model_type)
    if config is None:
        raise click.ClickException(f"No preset config found for '{model_type.value}'.")

    overrides = {}

    if device is not None:
        overrides["device"] = device
    if temperature is not None:
        overrides["temperature"] = temperature
    if max_new_tokens is not None:
        overrides["max_new_tokens"] = max_new_tokens

    return replace(config, **overrides) if overrides else config


@click.command()
@click.argument("model", type=click.Choice([m.value for m in OCRModelType]))
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu').")
@click.option("--temperature", type=float, default=None, help="Sampling temperature.")
@click.option(
    "--max-new-tokens",
    type=int,
    default=None,
    help="Maximum new tokens to generate.",
)
@click.option("--show-reasoning", is_flag=True, help="Show model reasoning output.")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save output to file.",
)
def main(
    model: str,
    image: Path,
    device: str | None,
    temperature: float | None,
    max_new_tokens: int | None,
    show_reasoning: bool,
    output: Path | None,
) -> None:
    model_type = OCRModelType(model)

    config = build_config(
        model_type=model_type,
        device=device,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    model_cls = MultiOCRFactory.get_ocr_model(ocr_model_type=model_type)
    if model_cls is None:
        raise click.ClickException(f"Model type '{model}' is not registered.")

    ocr_model = model_cls(config)

    click.echo(f"Loading model '{config.ocr_model_id}'...")
    ocr_model.load_model()

    click.echo(f"Processing '{image}'...")
    pil_image = Image.open(image)
    result = ocr_model.process_image(pil_image)

    if show_reasoning and result.reasoning:
        click.echo("\n--- Reasoning ---")
        click.echo(result.reasoning)
        click.echo("--- End Reasoning ---\n")

    click.echo("\n--- OCR Result ---")
    click.echo(result.text)

    if output:
        output.write_text(result.text)
        click.echo(f"\nResult saved to '{output}'")


if __name__ == "__main__":
    main()
