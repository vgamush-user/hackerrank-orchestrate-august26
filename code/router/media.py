"""
Multimodal media triage: turns image/voice messages into text + risk flags
the classifier can score, per the multimodal-media-triage skill.

Extraction is cached by media_id (not message_id) in media_cache.json, since
the same poster/voice-note is frequently reused across many recipients in
this dataset. This keeps the pipeline deterministic and avoids redundant
extraction work.

If a media_id shows up that is NOT in the cache, we fail soft: we don't
fabricate OCR/ASR content. We return an empty resolved_text with a low
extraction_confidence and an "uncached_media" flag, so the classifier can
treat it honestly as low-informativeness rather than silently as "no
content = low priority" (a pitfall the skill explicitly warns against).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "media_cache.json")


@dataclass
class MediaExtraction:
    media_id: Optional[str]
    media_type: Optional[str]  # "image" | "voice" | None
    resolved_text: str  # OCR text / ASR transcript / "" if none
    description: str = ""
    flags: List[str] = field(default_factory=list)
    extraction_confidence: float = 1.0  # 1.0 for plain text messages (nothing to extract)
    notes: str = ""


def load_media_cache(path: str = DEFAULT_CACHE_PATH) -> dict:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_media(msg: dict, media_cache: dict) -> MediaExtraction:
    """Resolve a single messages.csv row's media (if any) into text + flags."""
    media_type = (msg.get("media_type") or "").strip().lower()
    media_id = (msg.get("media_id") or "").strip() or None
    text = (msg.get("message_text") or "").strip()

    if not media_type:
        return MediaExtraction(
            media_id=None, media_type=None, resolved_text=text, extraction_confidence=1.0
        )

    entry = media_cache.get(media_id) if media_id else None
    if entry is None:
        return MediaExtraction(
            media_id=media_id,
            media_type=media_type,
            resolved_text=text,
            flags=["uncached_media"],
            extraction_confidence=0.2,
            notes="No cached OCR/ASR extraction found for this media_id.",
        )

    if media_type == "image":
        ocr = entry.get("ocr_text") or ""
        combined = "\n".join(x for x in [text, ocr] if x)
        return MediaExtraction(
            media_id=media_id,
            media_type="image",
            resolved_text=combined,
            description=entry.get("description", ""),
            flags=list(entry.get("flags", [])),
            extraction_confidence=float(entry.get("extraction_confidence", 0.7)),
            notes=entry.get("notes", ""),
        )

    if media_type == "voice":
        asr = entry.get("raw_asr_text") or ""
        return MediaExtraction(
            media_id=media_id,
            media_type="voice",
            resolved_text=asr,
            description=f"voice note, {entry.get('duration_sec', '?')}s",
            flags=list(entry.get("flags", [])),
            extraction_confidence=float(entry.get("extraction_confidence", 0.3)),
            notes=entry.get("notes", ""),
        )

    return MediaExtraction(media_id=media_id, media_type=media_type, resolved_text=text)
