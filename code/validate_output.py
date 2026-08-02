#!/usr/bin/env python3
"""
Validates dataset/output.csv against the submission contract in
problem_statement.md / AGENTS.md §6.2.

Usage: python code/validate_output.py [output.csv] [messages.csv] [message_history.csv]
"""
import sys

import pandas as pd

ACTIONS = {"notify", "digest", "mute"}
TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}
EXPECTED_COLS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def validate(output_path: str, messages_path: str, history_path: str) -> bool:
    out = pd.read_csv(output_path, dtype=str)
    msgs = pd.read_csv(messages_path, dtype=str)
    errors = []

    if list(out.columns) != EXPECTED_COLS:
        errors.append(f"Column mismatch: {list(out.columns)} != {EXPECTED_COLS}")

    expected_ids = set(msgs["message_id"])
    actual_ids = set(out["message_id"])
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        errors.append(f"{len(missing)} message_ids missing from output, e.g. {list(missing)[:5]}")
    if extra:
        errors.append(f"{len(extra)} unexpected message_ids in output, e.g. {list(extra)[:5]}")

    dupes = out["message_id"][out["message_id"].duplicated()]
    if len(dupes):
        errors.append(f"{len(dupes)} duplicate message_id rows, e.g. {dupes.tolist()[:5]}")

    bad_actions = out[~out["action"].isin(ACTIONS)]
    if len(bad_actions):
        errors.append(f"{len(bad_actions)} rows with invalid action, e.g. {bad_actions['action'].unique()[:5]}")

    bad_types = out[~out["message_type"].isin(TYPES)]
    if len(bad_types):
        errors.append(f"{len(bad_types)} rows with invalid message_type, e.g. {bad_types['message_type'].unique()[:5]}")

    conf = pd.to_numeric(out["confidence"], errors="coerce")
    bad_conf = out[(conf.isna()) | (conf < 0) | (conf > 1)]
    if len(bad_conf):
        errors.append(f"{len(bad_conf)} rows with invalid confidence")
    if (conf == 1.0).any():
        errors.append(f"{(conf == 1.0).sum()} row(s) with confidence == 1.0 (skill guidance: never output 1.0)")

    empty_reason = out[out["reason"].fillna("").str.strip() == ""]
    if len(empty_reason):
        errors.append(f"{len(empty_reason)} rows with empty reason")

    bad_evidence = out[out["evidence_message_ids"].isna() | (out["evidence_message_ids"].str.strip() == "")]
    if len(bad_evidence):
        errors.append(f"{len(bad_evidence)} rows with blank evidence_message_ids (use literal 'none', not blank)")

    history_ids = set(pd.read_csv(history_path, dtype=str)["message_id"])

    def evidence_ok(field: str) -> bool:
        if field == "none":
            return True
        ids = field.split(";")
        return all(i in history_ids for i in ids)

    bad_ev_ids = out[~out["evidence_message_ids"].fillna("none").apply(evidence_ok)]
    if len(bad_ev_ids):
        errors.append(
            f"{len(bad_ev_ids)} rows with evidence_message_ids not found in message_history.csv, "
            f"e.g. {bad_ev_ids[['message_id','evidence_message_ids']].head(3).values.tolist()}"
        )

    if errors:
        print(f"FAILED - {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        return False
    print(f"PASSED - {len(out)} rows, all checks clean.")
    return True


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/output.csv"
    messages_path = sys.argv[2] if len(sys.argv) > 2 else "dataset/messages.csv"
    history_path = sys.argv[3] if len(sys.argv) > 3 else "dataset/message_history.csv"
    ok = validate(output_path, messages_path, history_path)
    raise SystemExit(0 if ok else 1)
