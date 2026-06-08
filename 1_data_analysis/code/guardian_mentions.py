#!/usr/bin/env python3
"""
Analyze all LLM response JSON files for mentions of "The Guardian" news source.

For each mention, records:
- filename, region, politician, prompt_version, repetition number
- where the mention appears: reasoning, web_search, or output
- the original sentence containing the mention (for downstream sentiment analysis)

Results are saved to 1_data_analysis/results/guardian_mentions.csv
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_PATH = (
    Path(__file__).parent.parent.parent
    / "0_prompt_llms"
    / "result"
    / "trial_1_reason_url"
)
OUTPUT_PATH = Path(__file__).parent.parent / "results" / "guardian_mentions.csv"
REGIONS = ["US", "UK", "Europe"]

# Make politicians_data importable
_code_dir = Path(__file__).parent.parent.parent / "0_prompt_llms" / "code"
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

from politicians_data import POLITICIANS  # noqa: E402

# ---------------------------------------------------------------------------
# Guardian pattern
# Matches: "the guardian", "guardian" (any capitalisation), theguardian.com,
# guardian.co.uk, guardian.com
# ---------------------------------------------------------------------------

GUARDIAN_PATTERN = re.compile(
    r"(?:"
    r"(?:the\s+)?guardian"   # "guardian" or "the guardian"
    r")",
    re.IGNORECASE,
)

# Sentence splitter — split on sentence-ending punctuation + whitespace,
# or on paragraph breaks (double newlines)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# Matches a leading markdown link footnote at the start of a segment, e.g.:
#   ([domain.com](https://example.com/...))
#   [Title](https://example.com/...)
#   https://example.com/...
LEADING_URL = re.compile(
    r"^\s*"
    r"(?:\([^\]]*\]\([^\)]+\)\)?|https?://\S+|\[[^\]]*\]\([^\)]+\))"
    r"\s*\n?"  # optional trailing newline after the footnote
)

# Matches sentences that are *purely* a URL / markdown footnote with no other text
URL_ONLY = re.compile(
    r"^\s*"
    r"(?:\([^\]]*\]\([^\)]+\)\)?|https?://\S+|\[[^\]]*\]\([^\)]+\))"
    r"\s*$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sentences_with_guardian(text: str) -> list[str]:
    """Return every sentence in *text* that contains a Guardian mention.

    Leading markdown footnotes / bare URLs are stripped from each segment.
    Segments that are *only* a URL/footnote (no prose) are dropped entirely.
    """
    if not text:
        return []
    parts = SENTENCE_SPLIT.split(text.strip())
    results = []
    for s in parts:
        s = s.strip()
        # Strip any leading markdown footnote or bare URL
        s = LEADING_URL.sub("", s).strip()
        # Skip if nothing meaningful remains
        if not s:
            continue
        if GUARDIAN_PATTERN.search(s):
            results.append(s)
    return results


def parse_filename(stem: str) -> tuple[int | None, int | None]:
    """
    Parse prompt index and repetition number from a stem like
    'prompt3_repetition2_medium'.  Returns (None, None) on failure.
    """
    m = re.match(r"prompt(\d+)_repetition(\d+)", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def politician_and_version(prompt_index: int, region: str) -> tuple[str, str]:
    """
    Derive politician name and prompt version from the prompt index.

    Convention:
        politician_index = prompt_index // 2
        even prompt_index → "disregards"
        odd  prompt_index → "values"
    """
    politicians = POLITICIANS.get(region, [])
    pol_idx = prompt_index // 2
    if pol_idx >= len(politicians):
        return "unknown", "unknown"
    prompt_version = "disregards" if prompt_index % 2 == 0 else "values"
    return politicians[pol_idx], prompt_version


# ---------------------------------------------------------------------------
# Per-block extractors
# ---------------------------------------------------------------------------

def extract_reasoning(block: dict) -> list[str]:
    """Guardian-containing sentences from a reasoning block's summary."""
    hits = []
    for item in block.get("summary", []):
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        hits.extend(sentences_with_guardian(text))
    return hits


def extract_web_search(block: dict) -> list[str]:
    """
    Guardian mentions from a web_search_call block.
    Covers: search query and source titles (if present).
    URL-only hits are skipped — the domain match is enough to confirm the source.
    Each hit is returned as a descriptive string prefixed with [query] or [source].
    """
    action = block.get("action", {})
    hits = []

    query = action.get("query", "")
    if GUARDIAN_PATTERN.search(query):
        hits.append(f"[query] {query}")

    for source in action.get("sources", []):
        title = source.get("title", "")
        # Only record hits that have a title; skip URL-only matches
        if title and GUARDIAN_PATTERN.search(title):
            hits.append(f"[source] {title}")

    return hits


def extract_output(block: dict) -> list[str]:
    """
    Guardian-containing sentences from the final message block.
    Covers: main response text (sentence-split) and inline citation titles.
    URL-only citation hits are skipped.
    """
    hits = []
    for content_item in block.get("content", []):
        if not isinstance(content_item, dict):
            continue

        # Main response prose
        hits.extend(sentences_with_guardian(content_item.get("text", "")))

        # Inline citations — only record hits that have a title; skip URL-only matches
        for ann in content_item.get("annotations", []):
            title = ann.get("title", "")
            if title and GUARDIAN_PATTERN.search(title):
                hits.append(f"[citation] {title}")

    return hits


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_file(json_path: Path, region: str) -> list[dict]:
    """Return a list of row dicts — one per Guardian mention — for one JSON file."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: could not read {json_path.name}: {e}")
        return []

    stem = json_path.stem
    prompt_index, repetition = parse_filename(stem)
    if prompt_index is None:
        return []

    politician, prompt_version = politician_and_version(prompt_index, region)

    rows = []
    for block_idx, block in enumerate(data.get("output", [])):
        block_type = block.get("type")

        if block_type == "reasoning":
            location = "reasoning"
            hits = extract_reasoning(block)
        elif block_type == "web_search_call":
            location = "web_search"
            hits = extract_web_search(block)
        elif block_type == "message":
            location = "output"
            hits = extract_output(block)
        else:
            continue

        for sentence in hits:
            rows.append(
                {
                    "filename": stem,
                    "region": region,
                    "politician": politician,
                    "prompt_version": prompt_version,
                    "repetition": repetition,
                    "mention_location": location,
                    "block_index": block_idx,
                    "sentence": sentence,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []

    for region in REGIONS:
        region_path = BASE_PATH / region
        if not region_path.exists():
            print(f"Warning: region path not found: {region_path}")
            continue

        json_files = sorted(region_path.glob("prompt*_repetition*_*.json"))
        print(f"{region}: processing {len(json_files)} files …")

        for json_path in json_files:
            all_rows.extend(process_file(json_path, region))

    df = pd.DataFrame(
        all_rows,
        columns=[
            "filename",
            "region",
            "politician",
            "prompt_version",
            "repetition",
            "mention_location",
            "block_index",
            "sentence",
        ],
    )

    df = df.sort_values(
        ["region", "politician", "prompt_version", "repetition", "mention_location", "block_index"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nDone. {len(df)} Guardian mention rows saved to:\n  {OUTPUT_PATH}")

    if not df.empty:
        print(f"\nBreakdown by mention_location:")
        print(df["mention_location"].value_counts().to_string())
        print(f"\nBreakdown by region:")
        print(df["region"].value_counts().to_string())
        print(f"\nFiles with at least one mention: {df['filename'].nunique()}")


if __name__ == "__main__":
    main()
