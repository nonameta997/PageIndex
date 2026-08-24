"""Provider-free, sanitized preflight for PageIndex TOC candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import unicodedata
from pathlib import Path

from PyPDF2 import PdfReader

from .page_index import (
    NUMBERED_TOC_AGGREGATE_MAX_CHARS,
    NUMBERED_TOC_AGGREGATE_MAX_ENTRIES,
    NUMBERED_TOC_CHUNK_MAX_CHARS,
    NUMBERED_TOC_CHUNK_MAX_ENTRIES,
    STRUCTURAL_TOC_MIN_COVERAGE_RATIO,
    STRUCTURAL_TOC_MIN_ENTRIES,
    TOC_APPLICABILITY_NONE,
    TOC_APPLICABILITY_PRINTED_NUMBERED,
    TOC_APPLICABILITY_STRUCTURAL_NUMBERED,
    TOC_APPLICABILITY_UNUSABLE_CANDIDATE,
    _chunk_numbered_toc,
    _numbered_toc_entries,
    _structural_toc_entries,
    classify_toc_candidate,
)


PREFLIGHT_TOC_SCAN_MAX_PAGES = 24
PREFLIGHT_HASH_PREFIX_LENGTH = 16


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _structure_metrics(entries: list[dict]) -> dict:
    seen = set()
    previous = None
    unique = True
    source_order = True
    hierarchy = True
    for item in entries:
        structure = item["structure"]
        key = tuple(int(part) for part in structure.split("."))
        if structure in seen:
            unique = False
        if previous is not None and key <= previous:
            source_order = False
        if len(key) > 1:
            parent = ".".join(str(part) for part in key[:-1])
            if parent not in seen:
                hierarchy = False
        seen.add(structure)
        previous = key
    return {
        "unique_structures": unique,
        "source_order_valid": source_order,
        "hierarchy_valid": hierarchy,
    }


def analyze_toc_candidate(
    toc_content: str | None,
    *,
    page_index_given_in_toc: str = "no",
    toc_page_list: list[int] | None = None,
    object_sha256_prefix: str | None = None,
    object_size: int | None = None,
    page_count: int | None = None,
) -> dict:
    applicability, quality = classify_toc_candidate(
        toc_content,
        page_index_given_in_toc,
    )
    structural_entries, structural_source = _structural_toc_entries(toc_content)
    printed_entries = _numbered_toc_entries(toc_content)
    digest_entries = printed_entries or structural_entries
    digest_payload = [
        {
            "structure": item["structure"],
            "title": _normalize_text(item["title"]),
            **({"page": item["page"]} if "page" in item else {}),
        }
        for item in digest_entries
    ]
    title_sequence_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    chunks = []
    try:
        chunks = _chunk_numbered_toc(digest_entries) if digest_entries else []
    except Exception:
        chunks = []

    return {
        "object_sha256_prefix": object_sha256_prefix,
        "object_size": object_size,
        "page_count": page_count,
        "toc_page_list": list(toc_page_list or []),
        "toc_applicability": applicability,
        "selected_route": (
            "process_no_toc"
            if applicability
            in {TOC_APPLICABILITY_UNUSABLE_CANDIDATE, TOC_APPLICABILITY_NONE}
            else "process_toc_with_page_numbers"
            if applicability == TOC_APPLICABILITY_PRINTED_NUMBERED
            else "process_toc_no_page_numbers"
        ),
        "source_line_count": structural_source["source_line_count"],
        "structural_entry_count": len(structural_entries),
        "printed_entry_count": len(printed_entries),
        "parsed_coverage_ratio": structural_source["coverage_ratio"],
        **_structure_metrics(structural_entries),
        "chunk_count": len(chunks),
        "aggregate_character_count": sum(
            len(item["structure"]) + len(item["title"]) + 24
            for item in digest_entries
        ),
        "title_sequence_digest": title_sequence_digest,
        "candidate_rejection_reason": quality["candidate_rejection_reason"],
        "quality_limits": {
            "structural_minimum_entries": STRUCTURAL_TOC_MIN_ENTRIES,
            "structural_minimum_coverage_ratio": STRUCTURAL_TOC_MIN_COVERAGE_RATIO,
            "chunk_max_entries": NUMBERED_TOC_CHUNK_MAX_ENTRIES,
            "chunk_max_characters": NUMBERED_TOC_CHUNK_MAX_CHARS,
            "aggregate_max_entries": NUMBERED_TOC_AGGREGATE_MAX_ENTRIES,
            "aggregate_max_characters": NUMBERED_TOC_AGGREGATE_MAX_CHARS,
        },
        "corrected_path_applies": applicability
        in {
            TOC_APPLICABILITY_PRINTED_NUMBERED,
            TOC_APPLICABILITY_STRUCTURAL_NUMBERED,
            TOC_APPLICABILITY_UNUSABLE_CANDIDATE,
            TOC_APPLICABILITY_NONE,
        },
    }


def analyze_pdf_bytes(pdf_bytes: bytes) -> dict:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    candidate_pages = []
    for page_index, page_text in enumerate(
        page_texts[:PREFLIGHT_TOC_SCAN_MAX_PAGES]
    ):
        entries, metrics = _structural_toc_entries(page_text)
        if entries and metrics["coverage_ratio"] >= STRUCTURAL_TOC_MIN_COVERAGE_RATIO:
            candidate_pages.append(page_index)
    candidate_text = "\n".join(page_texts[index] for index in candidate_pages)
    return analyze_toc_candidate(
        candidate_text or None,
        toc_page_list=candidate_pages,
        object_sha256_prefix=hashlib.sha256(pdf_bytes).hexdigest()[
            :PREFLIGHT_HASH_PREFIX_LENGTH
        ],
        object_size=len(pdf_bytes),
        page_count=len(page_texts),
    )


def _artifact_record(payload) -> tuple[dict, int | None]:
    if isinstance(payload, dict) and "toc_content" in payload:
        return payload, payload.get("total_page_number")
    if not isinstance(payload, list):
        raise ValueError("artifact must be a record or diagnostic record list")
    total_page_number = next(
        (
            item.get("total_page_number")
            for item in payload
            if isinstance(item, dict) and "total_page_number" in item
        ),
        None,
    )
    record = next(
        (
            item
            for item in payload
            if isinstance(item, dict) and "toc_content" in item
        ),
        None,
    )
    if record is None:
        raise ValueError("artifact has no TOC candidate record")
    return record, total_page_number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path)
    parser.add_argument("--artifact-json", type=str)
    parser.add_argument("--object-sha256-prefix")
    parser.add_argument("--object-size", type=int)
    args = parser.parse_args()
    if bool(args.pdf) == bool(args.artifact_json):
        parser.error("provide exactly one PDF or --artifact-json")

    if args.pdf:
        result = analyze_pdf_bytes(args.pdf.read_bytes())
    else:
        raw = (
            sys.stdin.buffer.read()
            if args.artifact_json == "-"
            else Path(args.artifact_json).read_bytes()
        )
        record, page_count = _artifact_record(json.loads(raw))
        result = analyze_toc_candidate(
            record.get("toc_content"),
            page_index_given_in_toc=record.get("page_index_given_in_toc", "no"),
            toc_page_list=record.get("toc_page_list") or [],
            object_sha256_prefix=args.object_sha256_prefix,
            object_size=args.object_size,
            page_count=page_count,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
