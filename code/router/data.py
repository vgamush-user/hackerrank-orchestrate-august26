"""
Data loading and indexing for the Message Notification Router.

Everything here is pure I/O + indexing (no scoring logic), so the rest of the
pipeline can do O(1)/O(log n) lookups instead of re-scanning the CSVs per
message. Per AGENTS.md / problem_statement.md this only ever reads files
under dataset/ -- no organizer-only files, no hardcoded labels.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Data:
    dataset_dir: str

    messages: List[dict] = field(default_factory=list)
    users: List[dict] = field(default_factory=list)
    groups: List[dict] = field(default_factory=list)
    group_members: List[dict] = field(default_factory=list)
    business_accounts: List[dict] = field(default_factory=list)
    user_business_history: List[dict] = field(default_factory=list)
    message_history: List[dict] = field(default_factory=list)
    message_events: List[dict] = field(default_factory=list)
    images: List[dict] = field(default_factory=list)
    voice_notes: List[dict] = field(default_factory=list)
    daily_notification_summary: List[dict] = field(default_factory=list)

    # ---- indices, built once ----
    users_by_id: Dict[str, dict] = field(default_factory=dict)
    groups_by_id: Dict[str, dict] = field(default_factory=dict)
    group_member_by_pair: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    business_by_id: Dict[str, dict] = field(default_factory=dict)
    user_business_by_pair: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    images_by_id: Dict[str, dict] = field(default_factory=dict)
    voice_by_id: Dict[str, dict] = field(default_factory=dict)

    history_by_id: Dict[str, dict] = field(default_factory=dict)
    history_by_user: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    events_by_user_message: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    daily_summary_by_pair: Dict[Tuple[str, str], dict] = field(default_factory=dict)


def load_data(dataset_dir: str) -> Data:
    d = Data(dataset_dir=dataset_dir)

    d.messages = _read_csv(os.path.join(dataset_dir, "messages.csv"))
    d.users = _read_csv(os.path.join(dataset_dir, "users.csv"))
    d.groups = _read_csv(os.path.join(dataset_dir, "groups.csv"))
    d.group_members = _read_csv(os.path.join(dataset_dir, "group_members.csv"))
    d.business_accounts = _read_csv(os.path.join(dataset_dir, "business_accounts.csv"))
    d.user_business_history = _read_csv(os.path.join(dataset_dir, "user_business_history.csv"))
    d.message_history = _read_csv(os.path.join(dataset_dir, "message_history.csv"))
    d.message_events = _read_csv(os.path.join(dataset_dir, "message_events.csv"))
    d.images = _read_csv(os.path.join(dataset_dir, "images.csv"))
    d.voice_notes = _read_csv(os.path.join(dataset_dir, "voice_notes.csv"))
    d.daily_notification_summary = _read_csv(os.path.join(dataset_dir, "daily_notification_summary.csv"))

    d.users_by_id = {r["user_id"]: r for r in d.users}
    d.groups_by_id = {r["group_id"]: r for r in d.groups}
    d.group_member_by_pair = {(r["group_id"], r["user_id"]): r for r in d.group_members}
    d.business_by_id = {r["business_id"]: r for r in d.business_accounts}
    d.user_business_by_pair = {(r["user_id"], r["business_id"]): r for r in d.user_business_history}
    d.images_by_id = {r["image_id"]: r for r in d.images}
    d.voice_by_id = {r["voice_note_id"]: r for r in d.voice_notes}

    for r in d.message_history:
        d.history_by_id[r["message_id"]] = r
        d.history_by_user[r["user_id"]].append(r)
    for uid in d.history_by_user:
        d.history_by_user[uid].sort(key=lambda r: r.get("created_at", ""), reverse=True)

    for r in d.message_events:
        d.events_by_user_message[(r["user_id"], r["message_id"])] = r

    for r in d.daily_notification_summary:
        d.daily_summary_by_pair[(r["user_id"], r["date"])] = r

    return d


def get_group_member(d: Data, group_id: Optional[str], user_id: str) -> Optional[dict]:
    if not group_id:
        return None
    return d.group_member_by_pair.get((group_id, user_id))


def get_user_business(d: Data, user_id: str, business_id: Optional[str]) -> Optional[dict]:
    if not business_id:
        return None
    return d.user_business_by_pair.get((user_id, business_id))


def get_daily_summary(d: Data, user_id: str, date: str) -> Optional[dict]:
    return d.daily_summary_by_pair.get((user_id, date))
