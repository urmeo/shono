"""Convert a fine-tuned Whisper checkpoint to CTranslate2 for fast inference.

faster-whisper runs CTranslate2 models, so the fine-tune is converted once with
``ct2-transformers-converter``. CTranslate2 needs CUDA 12 / cuDNN 9; on the older
cuDNN-8 Kaggle images, pin ``ctranslate2==4.4.0`` or conversion/inference fails.
The import is lazy — this module loads without ctranslate2 installed.
"""

from __future__ import annotations

from pathlib import Path


def to_ct2(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    quantization: str = "float16",
    force: bool = False,
) -> str:
    """Convert the HF Whisper model at ``model_dir`` into a CTranslate2 model.

    ``quantization`` is typically ``float16`` for a T4. Returns the output path.
    """
    from ctranslate2.converters import TransformersConverter

    converter = TransformersConverter(str(model_dir))
    converter.convert(str(output_dir), quantization=quantization, force=force)
    return str(output_dir)
