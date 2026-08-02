#!/usr/bin/env python3
"""
Self-evaluation harness: runs the router pipeline against
dataset/sample_messages.csv (solved examples) and reports agreement with the
provided action/message_type labels. This is a sanity check only -- per
problem_statement.md, sample_messages.csv exists "to understand the expected
output format and style", not as training data, so nothing here feeds back
into the classifier's logic.

Usage: python code/evaluate.py [--dataset-dir DATASET_DIR]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
    args = parser.parse_args()

    d = load_data(args.dataset_dir)
    cache = load_media_cache()

    sample_path = os.path.join(args.dataset_dir, "sample_messages.csv")
    with open(sample_path, newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    action_correct = 0
    type_correct = 0
    total = 0
    mismatches = []

    for row in samples:
        expected_action = row.get("action", "").strip()
        if not expected_action:
            continue  # unsolved sample row
        total += 1
        media = resolve_media(row, cache)
        decision = classify_message(d, row, media)

        a_ok = decision.action == expected_action
        t_ok = decision.message_type == row.get("message_type", "").strip()
        action_correct += a_ok
        type_correct += t_ok
        if not (a_ok and t_ok):
            mismatches.append((row["message_id"], expected_action, decision.action,
                                row.get("message_type"), decision.message_type))

    print(f"Solved samples evaluated: {total}")
    if total:
        print(f"action accuracy:       {action_correct}/{total} = {action_correct/total:.1%}")
        print(f"message_type accuracy: {type_correct}/{total} = {type_correct/total:.1%}")
    if mismatches:
        print("\nMismatches (message_id, expected_action, got_action, expected_type, got_type):")
        for m in mismatches:
            print(" ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
