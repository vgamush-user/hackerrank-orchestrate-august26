"""
Evidence retrieval: selects 0-3 historical message_ids from message_history.csv
that best support the current routing decision, per the evidence-retrieval
skill.

Approach (skill-mandated order): filter first (same user_id AND matching
sender_user_id/business_id/group_id), rank second (content similarity via
TF-IDF + recency + presence of a recorded event), then join message_events
for the actual outcome. Cross-user history is never used. Weak/no matches
return "none" rather than a padded or fuzzy match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Data

MAX_EVIDENCE = 3
MIN_SIMILARITY_FOR_CONTENT_MATCH = 0.12


@dataclass
class EvidenceItem:
    message_id: str
    similarity: float
    has_event: bool
    event_summary: str  # e.g. "dismissed", "opened+replied", "reported", "muted_after"


def _event_summary(d: Data, user_id: str, message_id: str) -> Optional[str]:
    ev = d.events_by_user_message.get((user_id, message_id))
    if not ev:
        return None
    parts = []
    if str(ev.get("message_opened")) == "1":
        parts.append("opened")
    if str(ev.get("message_replied")) == "1":
        parts.append("replied")
    if str(ev.get("notification_dismissed")) == "1":
        parts.append("dismissed")
    if str(ev.get("muted_after_message")) == "1":
        parts.append("muted_after")
    if str(ev.get("message_reported")) == "1":
        parts.append("reported")
    return "+".join(parts) if parts else "no_reaction_recorded"


def find_candidates(d: Data, msg: dict) -> List[dict]:
    """Filter step: same user, and same sender_user_id OR business_id OR group_id."""
    user_id = msg["user_id"]
    sender_user_id = (msg.get("sender_user_id") or "").strip() or None
    business_id = (msg.get("business_id") or "").strip() or None
    group_id = (msg.get("group_id") or "").strip() or None

    candidates = []
    for h in d.history_by_user.get(user_id, []):
        same_sender = sender_user_id and h.get("sender_user_id") == sender_user_id
        same_business = business_id and h.get("business_id") == business_id
        same_group = group_id and h.get("group_id") == group_id
        if same_sender or same_business or same_group:
            candidates.append(h)
    return candidates


def rank_evidence(d: Data, msg: dict, resolved_text: str, candidates: List[dict]) -> List[EvidenceItem]:
    if not candidates:
        return []

    user_id = msg["user_id"]
    corpus = [resolved_text or ""] + [c.get("message_text") or "" for c in candidates]

    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(corpus)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    except ValueError:
        # e.g. empty vocabulary (all-empty text, common for voice notes) --
        # fall back to zero similarity, ranking purely by recency/event.
        sims = [0.0] * len(candidates)

    items = []
    for c, sim in zip(candidates, sims):
        summary = _event_summary(d, user_id, c["message_id"])
        items.append(
            EvidenceItem(
                message_id=c["message_id"],
                similarity=float(sim),
                has_event=summary is not None,
                event_summary=summary or "",
            )
        )

    # Rank: has_event first (informative > merely similar), then similarity,
    # then recency (candidates list is already newest-first per user).
    items.sort(key=lambda it: (it.has_event, it.similarity), reverse=True)
    return items


def select_evidence(d: Data, msg: dict, resolved_text: str) -> List[EvidenceItem]:
    candidates = find_candidates(d, msg)
    ranked = rank_evidence(d, msg, resolved_text, candidates)

    selected: List[EvidenceItem] = []
    for it in ranked:
        # Keep evidence that either has a recorded event (informative about
        # outcome) OR is a strong content match -- skip weak, event-less,
        # low-similarity matches rather than padding the list.
        if it.has_event or it.similarity >= MIN_SIMILARITY_FOR_CONTENT_MATCH:
            selected.append(it)
        if len(selected) >= MAX_EVIDENCE:
            break

    return selected


def evidence_ids_field(items: List[EvidenceItem]) -> str:
    if not items:
        return "none"
    return ";".join(it.message_id for it in items)
