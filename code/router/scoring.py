"""
Axis scoring: usefulness, urgency, repetition, risk -- per message-router-
classifier skill. Deliberately rule-based/deterministic (no external LLM
call) so the submission is runnable offline, reproducible, and needs no API
key, per the AGENTS.md project contract ("Keep behavior deterministic where
possible", "read secrets from environment variables only").

Each score is accompanied by a short list of concrete reasons so `reason`
strings in the final output cite real signals instead of vibes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from .data import Data, get_daily_summary, get_group_member, get_user_business

# ---------------------------------------------------------------------------
# Keyword banks
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = re.compile(
    r"routing override|system note for|assistant instruction|notification router|"
    r"mark (this|notify)|classify (this )?as|set action\s*=|ignore sender risk|"
    r"always mark this as|treat this as (notify|urgent)",
    re.IGNORECASE,
)

CREDENTIAL_REQUEST = re.compile(
    r"\botp\b|verify (your )?(account|wallet)|confirm your (wallet )?pin|login code|"
    r"\bpin\b.*(confirm|verify|share)|share the (code|otp)|enter (the )?otp|"
    r"send (the )?(code|otp) here|\d\s?digit (login )?code|cvv|"
    r"verify (now|immediately|at )|security alert|"
    r"(account|profile|access|wallet)s? (may|will) be (temporarily )?(blocked|suspended|deactivated)|"
    r"account will expire|access will expire|reply with the .*code|"
    r"verify (wallet|card|account) (and|or)?.*details|(wallet|card|account) details before",
    re.IGNORECASE,
)

PAYMENT_PRESSURE = re.compile(
    r"pay (a |the |small )?(re)?attempt(ing)? fee|pay .*(token|fee) today|"
    r"token .* (today|now) to block|complete before \d|payment .*(pending|failed)|"
    r"wallet pin|reactivation fee|clear(ed)? now|pay before \d",
    re.IGNORECASE,
)

URGENCY_WORDS = re.compile(
    r"\basap\b|\burgent\b|\bemergency\b|immediately|right now|"
    r"\b(pay|verify|reply|call|confirm|act|respond|scan|fill|come online)\s+now\b|before midnight|"
    r"before \d{1,2}(:\d{2})?\s*(am|pm)?|by \d{1,2}(:\d{2})?\s*(am|pm)?|today only|closes today|"
    r"closes at|\bdeadline\b|don'?t (wait|delay)|expire(s)?\s*(today|soon|now)?|will expire|"
    r"last chance|\beod\b|end of day|in \d+\s*min(ute)?s?|\bescalat|scheduled time|"
    r"within \d+\s*(min|hour)|max\b.*stop|last (call|chance)|"
    r"tomorrow morning|this evening|closes (this |tomorrow )?(evening|morning)|"
    r"expected to reach.*today|out for delivery|arriving today|leaving \d+\s*min(ute)?s? early|"
    r"clos(e|es|ing) (at|today)|portal (locks|closes)|won'?t be accepted|late entries|"
    r"before .* locks|refund processing will close",
    re.IGNORECASE,
)

NEGATED_URGENCY = re.compile(
    r"nothing urgent|no rush|not urgent|no hurry|whenever (works|you can|is fine)|"
    r"take your time|not (that )?urgent",
    re.IGNORECASE,
)

SUSPICIOUS_LINK = re.compile(
    r"[a-z0-9.-]+\.(in|com|sg|pro)\b.*(pay|kyc|verify|secure|delivery)|"
    r"pay-check-secure|amazonpay-delivery|hdfcbank-kyc|scan this qr|click here|use this link",
    re.IGNORECASE,
)

CHAIN_FORWARD = re.compile(
    r"forward (this |to )?(to )?(at least )?\d+ people|don'?t (break the chain|ignore)|"
    r"send to all (family )?groups|share in all family groups|do not ignore|"
    r"pls forward|please forward|forward (it |this )?to (family|group)|fwd as received|"
    r"sharing here in case it helps",
    re.IGNORECASE,
)

DIRECT_ASK = re.compile(
    r"\bcan you\b|\bplease\b|\breply\b|\bconfirm\b|let me know|need (your|you to)|"
    r"can you collect|rsvp|kindly|would you", re.IGNORECASE,
)

UNSUBSCRIBE = re.compile(r"reply stop|unsubscribe", re.IGNORECASE)

REPORT_THRESHOLD = 0.02  # user_reports_30d / messages_sent_30d above this is "high" report rate
NEW_ACCOUNT_DAYS = 90


def mentions_user(text: str, user_id: str) -> bool:
    """Detect an explicit @user_id mention of the receiving user -- per the
    problem statement, a muted/passive group can still contain an urgent
    direct mention of this specific user, which should override the
    group's usual quiet default."""
    if not text or not user_id:
        return False
    return bool(re.search(rf"@{re.escape(user_id)}\b", text))


@dataclass
class AxisScores:
    usefulness: float = 0.0       # -1 (actively unwanted) .. 0 (neutral) .. 1 (clearly useful)
    urgency: float = 0.0          # 0 .. 1
    risk: float = 0.0             # 0 .. 1
    repetition_ignore: float = 0.0  # 0 .. 1, how strongly history says "user ignores this"
    repetition_support: int = 0     # how many historical events back repetition_ignore

    injection_detected: bool = False
    scam_credential_or_payment_ask: bool = False

    usefulness_reasons: List[str] = field(default_factory=list)
    urgency_reasons: List[str] = field(default_factory=list)
    risk_reasons: List[str] = field(default_factory=list)
    repetition_reasons: List[str] = field(default_factory=list)


def _report_rate(business: dict) -> float:
    try:
        sent = float(business.get("messages_sent_30d") or 0)
        reports = float(business.get("user_reports_30d") or 0)
        return reports / sent if sent > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def score_business_risk(business: Optional[dict]) -> Tuple[float, List[str]]:
    if not business:
        return 0.0, []
    score = 0.0
    reasons = []

    verified = str(business.get("verified", "0")) == "1"
    if not verified:
        score += 0.35
        reasons.append(f"business account '{business.get('display_name')}' is not verified")

    official = (business.get("official_domain") or "").strip().lower()
    used = (business.get("domain_used_by_sender") or "").strip().lower()
    if official and used and official != used:
        score += 0.4
        reasons.append(
            f"sender domain '{used}' does not match {business.get('display_name')}'s official domain '{official}'"
        )

    try:
        age = float(business.get("account_age_days") or 0)
        if age < NEW_ACCOUNT_DAYS:
            score += 0.2
            reasons.append(f"business account is only {int(age)} days old")
    except (TypeError, ValueError):
        pass

    rate = _report_rate(business)
    if rate > REPORT_THRESHOLD:
        score += 0.25
        reasons.append(f"business has an elevated report rate ({rate:.1%} of recent messages)")

    return min(score, 1.0), reasons


def score_message(
    d: Data,
    msg: dict,
    resolved_text: str,
    media_flags: List[str],
    media_extraction_confidence: float,
) -> AxisScores:
    s = AxisScores()
    text = resolved_text or ""
    user_id = msg["user_id"]
    conv_type = msg.get("conversation_type", "")
    business_id = (msg.get("business_id") or "").strip() or None
    group_id = (msg.get("group_id") or "").strip() or None
    sender_user_id = (msg.get("sender_user_id") or "").strip() or None

    business = d.business_by_id.get(business_id) if business_id else None
    group = d.groups_by_id.get(group_id) if group_id else None
    member = get_group_member(d, group_id, user_id)
    ub = get_user_business(d, user_id, business_id)

    # --- injection detection -------------------------------------------------
    if INJECTION_PATTERNS.search(text):
        s.injection_detected = True
        s.risk_reasons.append("message text attempts to instruct the router directly (prompt-injection pattern)")

    # --- risk axis -------------------------------------------------------------
    risk = 0.0
    cred_hit = bool(CREDENTIAL_REQUEST.search(text))
    pay_hit = bool(PAYMENT_PRESSURE.search(text))
    link_hit = bool(SUSPICIOUS_LINK.search(text))
    urgent_hit = bool(URGENCY_WORDS.search(text)) and not bool(NEGATED_URGENCY.search(text))

    if cred_hit:
        risk += 0.45
        s.risk_reasons.append("message asks for OTP/PIN/login code or account verification")
    if pay_hit:
        risk += 0.3
        s.risk_reasons.append("message pressures an urgent/upfront payment")
    if link_hit:
        risk += 0.2
        s.risk_reasons.append("message pushes a suspicious link/QR/domain")
    if (cred_hit or pay_hit) and urgent_hit:
        risk += 0.15
        s.risk_reasons.append("urgency is combined with a payment/credential ask (classic scam combo)")
    if s.injection_detected and (cred_hit or pay_hit or link_hit):
        risk += 0.2
        s.scam_credential_or_payment_ask = True

    biz_risk, biz_reasons = score_business_risk(business)
    if biz_risk > 0:
        risk += biz_risk * 0.6
        s.risk_reasons.extend(biz_reasons)

    deceptive_media = False
    for flag in media_flags:
        if "brand_mismatch" in flag:
            risk += 0.2
            s.risk_reasons.append(f"media flag: {flag.replace('_', ' ')}")
            deceptive_media = True
        elif flag in ("fake_or_generic_contact_details", "placeholder_template_flyer", "advance_payment_request_in_caption"):
            risk += 0.15
            s.risk_reasons.append(f"media flag: {flag.replace('_', ' ')}")
            deceptive_media = True

    # deceptive identity (fake/impersonating business, mismatched brand, or
    # a placeholder/template flyer) combined with a payment/credential ask
    # is a scam pattern even without an explicit prompt-injection attempt.
    deceptive_identity = s.injection_detected or biz_risk >= 0.5 or deceptive_media
    if deceptive_identity and (cred_hit or pay_hit):
        s.scam_credential_or_payment_ask = True

    # a personal-conversation sender with no prior history to this user who
    # immediately asks for credentials/payment is a classic cold-contact
    # scam pattern, even with no business account or injection text to key
    # off of.
    if conv_type == "personal" and (cred_hit or pay_hit):
        prior_from_sender = any(
            h.get("sender_user_id") == sender_user_id for h in d.history_by_user.get(user_id, [])
        ) if sender_user_id else False
        if not prior_from_sender:
            risk += 0.25
            s.scam_credential_or_payment_ask = True
            s.risk_reasons.append("first message from this sender and it asks for sensitive verification or payment")

    # a business account that is both unverified AND uses a domain that
    # doesn't match its declared official domain is, on its own, a strong
    # brand-impersonation signal (e.g. a fake "<Bank> Helpdesk" account
    # using a *-kyc.in lookalike domain) -- treat as scam-grade even if the
    # message content itself couldn't be reliably read (e.g. garbled ASR).
    if business is not None:
        verified = str(business.get("verified", "0")) == "1"
        official = (business.get("official_domain") or "").strip().lower()
        used = (business.get("domain_used_by_sender") or "").strip().lower()
        domain_mismatch = bool(official and used and official != used)
        if not verified and domain_mismatch:
            s.scam_credential_or_payment_ask = True

    if CHAIN_FORWARD.search(text):
        risk += 0.1
        s.risk_reasons.append("classic forward-chain / good-luck-chain-letter pattern")

    try:
        fwd = int(float(msg.get("forwarded_count") or 0))
    except (TypeError, ValueError):
        fwd = 0
    if fwd >= 7:
        risk += 0.1
        s.risk_reasons.append(f"forwarded {fwd} times, a common scam/spam propagation signal")

    s.risk = min(risk, 1.0)

    # --- urgency axis ------------------------------------------------------
    urgency = 0.0
    if urgent_hit:
        urgency += 0.45
        s.urgency_reasons.append("contains explicit time-pressure language")
    if cred_hit:
        urgency += 0.2  # urgent-feeling even if it's a scam; risk axis will dominate downstream
    if conv_type == "personal":
        urgency += 0.15
        s.urgency_reasons.append("direct personal message")
    if DIRECT_ASK.search(text) and conv_type != "business":
        urgency += 0.2
        s.urgency_reasons.append("sender directly asks this user for a reply/action")
    if member and str(member.get("role")) == "admin":
        urgency += 0.1
        s.urgency_reasons.append("sent by a group admin")
    if mentions_user(text, user_id):
        urgency += 0.35
        s.urgency_reasons.append("directly @mentions this user, overriding the group's usual quiet default")
    s.urgency = min(urgency, 1.0)

    # --- usefulness axis -----------------------------------------------------
    usefulness = 0.0
    if ub:
        allows = str(ub.get("allows_promotions", "0")) == "1"
        opted_out = bool((ub.get("promotions_opted_out_at") or "").strip())
        activity = 0
        try:
            activity = int(float(ub.get("activity_count_180d") or 0))
        except (TypeError, ValueError):
            pass
        if opted_out:
            usefulness -= 0.5
            s.usefulness_reasons.append("user opted out of promotions from this business")
        elif allows and activity > 0:
            usefulness += 0.35
            s.usefulness_reasons.append("user has an active relationship/opt-in with this business")
        elif activity > 0:
            usefulness += 0.15
            s.usefulness_reasons.append("user has recent activity history with this business")
    if member:
        try:
            dismissed = int(float(member.get("notifications_dismissed_30d") or 0))
            replies = int(float(member.get("replies_sent_30d") or 0))
        except (TypeError, ValueError):
            dismissed, replies = 0, 0
        if str(member.get("group_muted_by_user")) == "1":
            usefulness -= 0.3
            s.usefulness_reasons.append("user has muted this group")
        if replies > 0:
            usefulness += 0.1
    if DIRECT_ASK.search(text):
        usefulness += 0.15
        s.usefulness_reasons.append("message asks the user for something specific")
    if UNSUBSCRIBE.search(text) and not ub:
        usefulness -= 0.1
        s.usefulness_reasons.append("unsolicited marketing message with no user relationship on file")

    s.usefulness = max(-1.0, min(usefulness, 1.0))

    return s


def is_in_dnd_window(dnd_window: str, created_at: str) -> bool:
    """Best-effort check whether created_at's time-of-day falls inside the
    user's do_not_disturb_window (format 'HH:MM-HH:MM', possibly wrapping
    past midnight)."""
    try:
        start_s, end_s = dnd_window.split("-")
        start = datetime.strptime(start_s.strip(), "%H:%M").time()
        end = datetime.strptime(end_s.strip(), "%H:%M").time()
        t = datetime.strptime(created_at.split(" ")[1], "%H:%M").time()
    except Exception:
        return False
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def is_notification_saturated(d: Data, user_id: str, date: str, threshold: int = 6) -> bool:
    summary = get_daily_summary(d, user_id, date)
    if not summary:
        return False
    try:
        sent = int(float(summary.get("notifications_sent") or 0))
    except (TypeError, ValueError):
        return False
    return sent >= threshold
