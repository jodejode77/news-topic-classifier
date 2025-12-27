"""Text preprocessing utilities."""
from __future__ import annotations

from typing import Optional


def build_text(headline: str, short_description: Optional[str], separator: str) -> str:
    """Combine headline and short_description into a single text field."""
    headline_clean = (headline or "").strip()
    description_clean = (short_description or "").strip()
    if headline_clean and description_clean:
        return f"{headline_clean}{separator}{description_clean}"
    if headline_clean:
        return headline_clean
    return description_clean
