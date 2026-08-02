"""
Final decision layer: combines the four axis scores (usefulness, urgency,
risk, repetition) into action + message_type, per message-router-classifier's
combination rule (risk dominates), writes a reason string that cites the
decisive signal, and calibrates confidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .data import Data, get_daily_summary, get_group_member
from .evidence import EvidenceItem, evidence_ids_field, select_evidence
from .media import MediaExtraction
from .scoring import (
    AxisScores,
    CREDENTIAL_REQUEST,
    PAYMENT_PRESSURE,
    URGENCY_WORDS,
    is_in_dnd_window,
    is_notification_saturated,
    score_message,
)

ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_TYPES = {
    "personal", "urgent", "event", "payment", "business_update", "promotion",
    "greeting", "forward", "spam", "scam", "unknown",
}

RISK_HIGH = 0.55
URGENCY_HIGH = 0.45
USEFULNESS_HIGH = 0.35
REPETITION_HIGH = 0.55
REPETITION_MIN_SUPPORT = 2

EVENT_WORDS = re.compile(
    r"\bfield trip\b|\bmeetup\b|\brsvp\b|\bmeeting\b|\bsync\b|\bworkshop\b|\bwalkathon\b|"
    r"\bconsent\b|\bfire alarm\b|\bmaintenance\b|\btrip\b|\bschedule\b|\bbus\b.*\b(leav|time)|"
    r"\bappointment\b|\breservation\b|\bbooking\b.*\b(update|time)|kids down by",
    re.IGNORECASE,
)
PAYMENT_WORDS = re.compile(
    r"payment due|\binvoice\b|\breceipt\b|\bbill\b|rent due|fee due|\bemi\b|recharge|dues pending",
    re.IGNORECASE,
)
PROMOTION_WORDS = re.compile(
    r"%\s?off|\bsale\b|\bdiscount\b|\bcashback\b|\boffer\b|\bdeal\b|book now|starting at|"
    r"limited time|prime day|price final",
    re.IGNORECASE,
)
GREETING_WORDS = re.compile(
    r"good morning|good night|\bblessing|best wishes|\bcongratulations\b|happy (diwali|new year|birthday)",
    re.IGNORECASE,
)
BUSINESS_UPDATE_WORDS = re.compile(
    r"\border\b|\bdelivery\b|\bbooking\b|account status|\bconfirmation\b|has been packed|"
    r"\bshipped\b|\bhub\b|\breturn\b.*pickup",
    re.IGNORECASE,
)


@dataclass
class Decision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str


def _pick_message_type(
    msg: dict, text: str, scores: AxisScores, action: str, media: MediaExtraction
) -> str:
    if action == "mute" and scores.risk >= RISK_HIGH:
        if (
            scores.scam_credential_or_payment_ask
            or scores.injection_detected
            or CREDENTIAL_REQUEST.search(text)
        ):
            return "scam"
        return "scam" if (scores.risk >= 0.7) else "spam"

    try:
        fwd = int(float(msg.get("forwarded_count") or 0))
    except (TypeError, ValueError):
        fwd = 0
    if fwd >= 5 and not BUSINESS_UPDATE_WORDS.search(text):
        return "forward"

    if action == "mute" and scores.repetition_ignore >= REPETITION_HIGH:
        return "spam"

    if EVENT_WORDS.search(text):
        return "event"
    if msg.get("conversation_type") == "business":
        if PROMOTION_WORDS.search(text) or "unsubscribe" in text.lower():
            return "promotion"
        return "business_update"
    if PAYMENT_WORDS.search(text):
        return "payment"
    if URGENCY_WORDS.search(text) and action == "notify":
        return "urgent"
    if PROMOTION_WORDS.search(text):
        return "promotion"
    if GREETING_WORDS.search(text):
        return "greeting"
    if msg.get("conversation_type") == "personal":
        return "personal"
    if not text.strip() or media.extraction_confidence < 0.3:
        return "unknown"
    return "personal"


def _compose_reason(
    scores: AxisScores, action: str, msg_type: str, evidence: List[EvidenceItem], media: MediaExtraction
) -> str:
    if action == "mute" and scores.risk >= RISK_HIGH:
        top = scores.risk_reasons[:2]
        base = "; ".join(top) if top else "content matches known scam/spam patterns"
        return base[0].upper() + base[1:] + "."

    if action == "mute" and scores.repetition_ignore >= REPETITION_HIGH:
        n = scores.repetition_support
        return (
            f"User has ignored/dismissed {n} similar past message(s) from this sender; "
            "no new signal justifies interrupting now."
        )

    if action == "notify" and scores.urgency >= URGENCY_HIGH:
        top = scores.urgency_reasons[:2]
        base = "; ".join(top) if top else "message carries genuine time pressure"
        return base[0].upper() + base[1:] + "."

    if action == "digest" and scores.usefulness >= USEFULNESS_HIGH:
        top = scores.usefulness_reasons[:2]
        base = "; ".join(top) if top else "message is useful but not time-sensitive"
        return base[0].upper() + base[1:] + "."

    if scores.usefulness_reasons or scores.urgency_reasons:
        top = (scores.usefulness_reasons + scores.urgency_reasons)[:2]
        base = "; ".join(top)
        return base[0].upper() + base[1:] + "."

    if media.description:
        return f"Low-signal media message ({media.description}); no urgency, strong usefulness, or risk detected."

    return "No strong urgency, usefulness, repetition, or risk signal found; routed by default fallback."


def _confidence(scores: AxisScores, action: str, evidence: List[EvidenceItem], media: MediaExtraction) -> float:
    if action == "mute" and scores.risk >= RISK_HIGH:
        base = 0.8 + min(scores.risk - RISK_HIGH, 0.2) * 0.75
    elif action == "mute" and scores.repetition_ignore >= REPETITION_HIGH:
        base = 0.75 + min(scores.repetition_support / 10, 0.15)
    elif action == "notify" and scores.urgency >= URGENCY_HIGH:
        base = 0.65 + min(scores.urgency - URGENCY_HIGH, 0.4) * 0.5
    elif action == "digest" and scores.usefulness >= USEFULNESS_HIGH:
        base = 0.65 + min(scores.usefulness - USEFULNESS_HIGH, 0.4) * 0.4
    else:
        base = 0.45

    if evidence and any(e.has_event for e in evidence):
        base += 0.03
    if media.media_type and media.extraction_confidence < 0.35:
        base -= 0.08

    return round(max(0.35, min(base, 0.95)), 2)


def classify_message(d: Data, msg: dict, media: MediaExtraction) -> Decision:
    text = media.resolved_text or ""
    scores = score_message(d, msg, text, media.flags, media.extraction_confidence)

    evidence = select_evidence(d, msg, text)
    ignored = sum(1 for e in evidence if any(k in e.event_summary for k in ("dismissed", "muted_after", "reported")))
    engaged = sum(1 for e in evidence if any(k in e.event_summary for k in ("opened", "replied")))
    support = ignored + engaged
    scores.repetition_support = support
    if support > 0:
        scores.repetition_ignore = ignored / support
        if ignored:
            scores.repetition_reasons.append(
                f"{ignored} of {support} similar past message(s) from this sender/business/group were dismissed, muted, or reported"
            )

    user = d.users_by_id.get(msg["user_id"], {})

    # --- combination rule (risk dominates) ---------------------------------
    if scores.risk >= RISK_HIGH:
        action = "mute"
    elif scores.urgency >= URGENCY_HIGH and scores.usefulness > -0.2:
        action = "notify"
    elif scores.usefulness >= USEFULNESS_HIGH:
        action = "digest"
    elif scores.repetition_ignore >= REPETITION_HIGH and scores.repetition_support >= REPETITION_MIN_SUPPORT:
        action = "mute"
    else:
        action = "digest" if scores.usefulness > -0.15 else "mute"

    if action == "notify" and scores.urgency < 0.75:
        date = (msg.get("created_at") or "").split(" ")[0]
        if is_notification_saturated(d, msg["user_id"], date):
            action = "digest"
            scores.usefulness_reasons.insert(0, "user already has a high notification volume today")

    msg_type = _pick_message_type(msg, text, scores, action, media)
    reason = _compose_reason(scores, action, msg_type, evidence, media)
    confidence = _confidence(scores, action, evidence, media)
    evidence_field = evidence_ids_field(evidence)

    assert action in ALLOWED_ACTIONS
    assert msg_type in ALLOWED_TYPES

    return Decision(
        message_id=msg["message_id"],
        action=action,
        message_type=msg_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids=evidence_field,
    )
