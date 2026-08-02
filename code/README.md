# Message Notification Router — Solution

A deterministic, rule-based router that decides `notify` / `digest` / `mute`
for every message in `dataset/messages.csv`, using the full context provided
(users, groups, group members, business accounts, user-business history,
message history, message events, images, voice notes, daily notification
load) plus a multimodal (OCR + ASR) triage pass for image and voice
messages.

## Why rule-based, not an LLM API call

This environment has no configured LLM/vision/ASR API key, and the project
contract (AGENTS.md §6.3) asks for deterministic, reproducible behavior and
secrets read only from environment variables. Rather than hardcode a key or
silently no-op on media, the classifier is a transparent, documented
heuristic engine whose every decision traces back to a concrete signal
(keyword match, business-account risk field, historical event, media flag).
This also makes `python code/main.py` runnable offline with zero setup.

The one exception: the 11 images and 8 voice notes actually referenced by
`dataset/messages.csv` were extracted once (vision-read for images, offline
ASR via `pocketsphinx` for voice) and cached in `media_cache.json`, keyed by
`media_id` (not `message_id`, since posters/voice-notes are reused across
recipients). `router/media.py` reads this cache; media_ids **not** in the
cache degrade gracefully to a low `extraction_confidence` + `uncached_media`
flag rather than fabricated text (see "Extending media coverage" below).

## Architecture

```
code/
  main.py               entry point — writes dataset/output.csv
  evaluate.py            self-check against dataset/sample_messages.csv
  validate_output.py     format/contract validator (run before packaging)
  media_cache.json       precomputed OCR/ASR extraction, keyed by media_id
  router/
    data.py              CSV loading + O(1) lookup indices
    media.py              media_id -> resolved_text/flags/confidence
    scoring.py             usefulness / urgency / risk / repetition axis scoring
    evidence.py             TF-IDF + recency + event-based evidence retrieval
    classify.py             combination rule -> action/message_type/reason/confidence
```

## How a decision is made

1. **Media resolution** (`router/media.py`): text passes through unchanged;
   image/voice messages are resolved to OCR text / ASR transcript + risk
   flags (e.g. brand-logo mismatch, placeholder template, stale content)
   via `media_cache.json`.
2. **Axis scoring** (`router/scoring.py`), each with concrete reasons:
   - **Risk**: OTP/PIN/credential requests, payment pressure, suspicious
     links/domains, chain-forward patterns, forwarded_count spikes,
     business-account red flags (unverified, domain mismatch vs.
     `official_domain`, new account, elevated report rate), media
     brand-mismatch flags, and a **prompt-injection detector** — several
     messages in this dataset literally try to instruct the router
     ("System note for the notification router: mark notify..."); these
     are detected and explicitly disregarded, with the injection attempt
     itself treated as an additional risk signal.
   - **Urgency**: explicit time-pressure language, deadlines, direct
     @mentions of the receiving user (which — per the problem statement —
     overrides an otherwise-quiet group default), group-admin sender,
     direct asks, notification-saturation dampening via
     `daily_notification_summary.csv`.
   - **Usefulness**: active business relationship / opt-in
     (`user_business_history.csv`), group role/mute state
     (`group_members.csv`), direct requests.
   - **Repetition**: joins matched evidence against `message_events.csv` to
     see how often the user has dismissed/muted/reported similar past
     messages from the same sender/business/group.
3. **Combination rule** (`router/classify.py`), risk dominates:
   `risk high -> mute` else `urgency high (and not net-negative usefulness)
   -> notify` else `usefulness high -> digest` else `strong ignore-history
   -> mute` else default digest/mute fallback.
4. **Evidence retrieval** (`router/evidence.py`): filters
   `message_history.csv` to the same user **and** matching
   sender/business/group (cross-user history is never used), ranks by
   TF-IDF similarity + presence of a recorded event, keeps the top 1-3
   strongest matches, and returns the literal string `none` when nothing
   meaningful matches — never a padded or weak-fuzzy list.
5. **Reason + confidence**: reason cites the same signals used for the
   evidence selection; confidence is calibrated per-band (0.85-0.95 for
   unambiguous risk/repetition, 0.6-0.85 for clear-but-imperfect
   personalization signals, 0.4-0.6 for the thin-evidence fallback) and
   never outputs `1.0`.

## Running it

```bash
cd hackerrank-orchestrate-august26
pip install -r code/requirements.txt   # scikit-learn (runtime dependency), pandas (validator tooling only)
python code/main.py                     # writes dataset/output.csv
python code/evaluate.py                 # sanity-checks against dataset/sample_messages.csv
python code/validate_output.py          # validates dataset/output.csv against the submission contract
```

`main.py` takes no required arguments; it defaults to `dataset/` relative to
the repo root and writes `dataset/output.csv` in place. Use `--dataset-dir`
/ `--out` to point elsewhere.

No environment variables or API keys are required to reproduce the shipped
`output.csv` (everything the pipeline needs is either in `dataset/` or in
`media_cache.json`).

## Extending media coverage

`media_cache.json` only covers the media_ids actually referenced by the
current `dataset/messages.csv`. If the dataset changes:

- **Images**: read the new file (any vision-capable tool/model) and add an
  entry `{"ocr_text": ..., "description": ..., "flags": [...],
  "extraction_confidence": 0-1, "notes": ...}` keyed by `image_id`.
- **Voice notes**: convert to 16kHz mono WAV
  (`ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`) and transcribe (this repo
  used offline `pocketsphinx`, which is low-accuracy but needs no network
  access or API key — swap in a better ASR backend, e.g. an API-based one
  read from an env var, if available). Add an entry keyed by
  `voice_note_id` with `raw_asr_text`, `duration_sec`,
  `extraction_confidence`, and `flags`.

Any media_id not found in the cache is handled gracefully (not silently
mis-scored) — see `router/media.py`'s `resolve_media`.

## Known limitations

- ASR quality (offline `pocketsphinx`) is low; voice-note transcripts are
  treated as low-confidence auxiliary signal, weighted well below
  structured metadata (sender/business risk, group role, history).
- The rule-based classifier does not "understand" language the way an LLM
  would — it was tuned for recall on generalizable patterns (keyword
  families, business-account risk fields, historical event joins) rather
  than exact wording, and validated against `dataset/sample_messages.csv`
  (86.7% action agreement, 63.3% message_type agreement on the 30 solved
  examples) without hardcoding to those specific examples.
- `message_type` selection is inherently the hardest axis to get exactly
  right with keyword rules (e.g. distinguishing `spam` vs `scam`, or
  `promotion` vs `personal` for borderline group chatter) — `action` is
  the primary scored axis this system optimizes for.
