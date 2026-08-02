#!/usr/bin/env python3
"""
Message Notification Router -- entry point.

Usage:
    python code/main.py [--dataset-dir DATASET_DIR] [--out OUT_CSV]

Reads dataset/messages.csv plus all provided context files (users, groups,
group_members, business_accounts, user_business_history, message_history,
message_events, images, voice_notes, daily_notification_summary) and writes
one prediction row per message to dataset/output.csv (or --out), with the
required columns:

    message_id,action,message_type,reason,confidence,evidence_message_ids

No network calls, no API keys, fully deterministic given the dataset and the
media_cache.json shipped alongside this code (see README.md for how that
cache was built and how to regenerate/extend it).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router.classify import classify_message  # noqa: E402
from router.data import load_data  # noqa: E402
from router.media import load_media_cache, resolve_media  # noqa: E402

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def run(dataset_dir: str, out_path: str, media_cache_path: str | None = None) -> int:
    d = load_data(dataset_dir)
    if not d.messages:
        print(f"No messages found under {dataset_dir}/messages.csv", file=sys.stderr)
        return 1

    cache = load_media_cache(media_cache_path) if media_cache_path else load_media_cache()

    rows = []
    for msg in d.messages:
        media = resolve_media(msg, cache)
        decision = classify_message(d, msg, media)
        rows.append(
            {
                "message_id": decision.message_id,
                "action": decision.action,
                "message_type": decision.message_type,
                "reason": decision.reason,
                "confidence": f"{decision.confidence:.2f}",
                "evidence_message_ids": decision.evidence_message_ids,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Message Notification Router")
    parser.add_argument("--dataset-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
    parser.add_argument("--out", default=None, help="defaults to <dataset-dir>/output.csv")
    parser.add_argument("--media-cache", default=None, help="defaults to code/media_cache.json")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.dataset_dir, "output.csv")
    return run(args.dataset_dir, out_path, args.media_cache)


if __name__ == "__main__":
    raise SystemExit(main())
