"""Measure where Claude translation output tokens go, on one real batch of a finished job.

Sends the SAME translation batch twice to the Anthropic Messages API:
  1. "legacy":  no `thinking` field (what SubBuild Studio sent before the fix -> adaptive thinking)
  2. "current": the request the app sends now (`anthropic_thinking_fields`, cached system prompt)
and prints `usage` plus the number of thinking blocks for each. Costs well under one cent.

Usage (from the backend folder, with the app's venv):
    python scripts/verify_translation_tokens.py job_320de88c99 [--model claude-haiku-5-5] [--start 0] [--size 18]
The API key comes from the app's key vault (the same one the pipeline uses).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from app.core.database import engine  # noqa: E402
from app.core.provider_keys import resolve_secret_key  # noqa: E402
from app.models.db_models import Job, SubtitleSegmentRecord  # noqa: E402
from app.providers.llm.client import (  # noqa: E402
    _anthropic_system_blocks,
    _schema_instruction,
    anthropic_thinking_fields,
)
from app.providers.translation.gemini_provider import (  # noqa: E402
    TranslationBatchWithFormResponseSchema,
    build_subtitle_system_prompt,
)


def _item(seg, *, meta=False, translated=False):
    it = {"id": seg.index}
    if meta:
        it["segment_type"] = seg.segment_type
        it["duration_sec"] = round(seg.end_time - seg.start_time, 3)
    it["original_text"] = seg.original_text
    if translated and seg.translated_text:
        it["translated_text"] = seg.translated_text
    return it


async def _send(api_key, payload):
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", type=int, default=18)
    args = ap.parse_args()

    with Session(engine) as session:
        job = session.get(Job, args.job_id)
        if job is None:
            sys.exit(f"Job {args.job_id} not found")
        segs = list(session.exec(
            select(SubtitleSegmentRecord).where(SubtitleSegmentRecord.job_id == args.job_id).order_by(SubtitleSegmentRecord.start_time)
        ))
        glossary = json.loads(job.glossary_json) if job.glossary_json else None

    batch = segs[args.start:args.start + args.size]
    prev = segs[max(0, args.start - 5):args.start]
    nxt = segs[args.start + args.size:args.start + args.size + 3]
    user = json.dumps({
        "previous_context": [_item(s, translated=True) for s in prev],
        "target_batch": [_item(s, meta=True) for s in batch],
        "lookahead_context": [_item(s) for s in nxt],
    }, ensure_ascii=False, separators=(",", ":"))
    system = build_subtitle_system_prompt("English", job.target_language or "Hebrew", glossary=glossary,
                                          report_addressee_form=True)
    schema = _schema_instruction(TranslationBatchWithFormResponseSchema)
    key = resolve_secret_key("anthropic")
    if key is None:
        sys.exit("No Anthropic key in the vault")

    base = {"model": args.model, "max_tokens": 8192, "messages": [{"role": "user", "content": user}]}
    legacy = dict(base, system=system + schema)
    current = dict(base, system=_anthropic_system_blocks(system, schema), **anthropic_thinking_fields(args.model))

    print(f"Batch of {len(batch)} lines; expected visible JSON ~{sum(len(s.translated_text or '') for s in batch) + 60 * len(batch)} chars")
    for label, payload in (("legacy (no thinking field)", legacy), ("current", current)):
        data = await _send(key.api_key, payload)
        thinking = sum(1 for b in data.get("content", []) if b.get("type") in ("thinking", "redacted_thinking"))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        print(f"\n== {label} ==\nusage: {json.dumps(data.get('usage'))}\nthinking blocks: {thinking}"
              f"\nvisible text: {len(text)} chars\nstop_reason: {data.get('stop_reason')}")


if __name__ == "__main__":
    asyncio.run(main())
