"""
formats/pdf_handler.py — PDF text and table extraction.

Uses pdfplumber as primary extractor (handles tables well), with pypdf
as a fallback for text-only extraction.

Public API
----------
PDFHandler()
    .extract(path: Path | str) -> pd.DataFrame
      Columns: page, content_type, content
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PDFHandler:
    """Extracts text and tables from PDF files using pdfplumber."""

    def extract(self, path: Path | str) -> pd.DataFrame:
        """
        Parse a PDF and return a DataFrame with columns:
        [page, content_type, content]
        """
        path = Path(path)
        try:
            return self._extract_pdfplumber(path)
        except ImportError:
            logger.warning("pdfplumber not available — falling back to pypdf.")
            return self._extract_pypdf(path)

    # ── Primary extractor ─────────────────────────────────────────────────────

    def _extract_pdfplumber(self, path: Path) -> pd.DataFrame:
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise ImportError("Install pdfplumber: pip install pdfplumber") from exc

        rows: list[dict] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Text blocks
                text = page.extract_text() or ""
                for block in text.split("\n\n"):
                    block = block.strip()
                    if block:
                        rows.append({"page": page_num, "content_type": "text", "content": block})

                # Tables
                for table in page.extract_tables():
                    if not table:
                        continue
                    headers = [str(h).strip() for h in table[0]]
                    for row in table[1:]:
                        row_data = {h: str(row[i]).strip() for i, h in enumerate(headers) if i < len(row)}
                        rows.append({
                            "page": page_num,
                            "content_type": "table_row",
                            "content": str(row_data),
                        })

        logger.info("PDF extracted (pdfplumber): %s → %d items", path.name, len(rows))
        return pd.DataFrame(rows, columns=["page", "content_type", "content"])

    # ── Fallback extractor ────────────────────────────────────────────────────

    def _extract_pypdf(self, path: Path) -> pd.DataFrame:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise ImportError("Install pypdf: pip install pypdf") from exc

        rows: list[dict] = []
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for block in text.split("\n\n"):
                block = block.strip()
                if block:
                    rows.append({"page": page_num, "content_type": "text", "content": block})

        logger.info("PDF extracted (pypdf fallback): %s → %d items", path.name, len(rows))
        return pd.DataFrame(rows, columns=["page", "content_type", "content"])
