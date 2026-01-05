import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Минимальная, предсказуемая нормализация.
    Никакой агрессивной чистки.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_batch(texts: List[str]) -> List[str]:
    return [normalize_text(t) for t in texts]
