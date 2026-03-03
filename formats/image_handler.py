"""
formats/image_handler.py — Screenshot and image OCR via Ollama llava.

For images: sends raw bytes to llava (multimodal Ollama model) with a
prompt to extract all visible text and tabular data, then returns results
as a DataFrame. Falls back to pytesseract for text-only OCR if llava
is unavailable.

Public API
----------
ImageHandler(llm_client=None)
    .extract(path: Path | str) -> pd.DataFrame
      Columns: source, content_type, content
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_LLAVA_PROMPT = (
    "Extract all text from this image exactly as it appears. "
    "If there are tables, reproduce them as CSV. "
    "If there is no useful text, say 'No text found'."
)


class ImageHandler:
    """
    Extracts text from images using Ollama llava (vision LLM) or
    pytesseract as a fallback.
    """

    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client   # optional LLMClient injected by app

    def extract(self, path: Path | str) -> pd.DataFrame:
        """Return a DataFrame with extracted text from the image."""
        path = Path(path)
        image_bytes = path.read_bytes()

        if self.llm_client is not None:
            return self._extract_llava(path, image_bytes)
        return self._extract_tesseract(path, image_bytes)

    # ── llava extraction ──────────────────────────────────────────────────────

    def _extract_llava(self, path: Path, image_bytes: bytes) -> pd.DataFrame:
        try:
            text = self.llm_client.vision_chat(
                messages=[{"role": "user", "content": _LLAVA_PROMPT}],
                image_bytes=image_bytes,
            )
            logger.info("llava extracted %d chars from %s", len(text), path.name)
        except Exception as exc:
            logger.warning("llava extraction failed (%s), trying tesseract.", exc)
            return self._extract_tesseract(path, image_bytes)

        return pd.DataFrame([{
            "source": path.name,
            "content_type": "ocr_llava",
            "content": text.strip(),
        }])

    # ── tesseract fallback ────────────────────────────────────────────────────

    def _extract_tesseract(self, path: Path, image_bytes: bytes) -> pd.DataFrame:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            import io

            img = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img)
            logger.info("tesseract extracted %d chars from %s", len(text), path.name)
            return pd.DataFrame([{
                "source": path.name,
                "content_type": "ocr_tesseract",
                "content": text.strip(),
            }])
        except ImportError as exc:
            raise ImportError(
                "No vision backend available. Install pytesseract + tesseract binary, "
                "or provide an LLMClient with llava model."
            ) from exc
