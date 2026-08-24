"""Provider-free, sanitized preflight for PageIndex TOC candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
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
    TocTransformationExhausted,
    _chunk_numbered_toc,
    _numbered_toc_entries,
    _structural_toc_entries,
    classify_toc_candidate,
    classify_toc_candidate_metrics,
)


PREFLIGHT_TOC_SCAN_MAX_PAGES = 24
PREFLIGHT_HASH_PREFIX_LENGTH = 16
HISTORICAL_EVIDENCE_MODE = "historical_detector_artifact_replay"
FRESH_SCAN_EVIDENCE_MODE = "fresh_raw_page_scan"

_HISTORICAL_ROOT_KEYS = {
    "schema_version",
    "evidence_mode",
    "source_object",
    "artifacts",
}
_HISTORICAL_SOURCE_KEYS = {
    "object_sha256_prefix",
    "object_size",
    "page_count",
}
_HISTORICAL_ARTIFACT_KEYS = {
    "observation_id",
    "artifact_sha256",
    "artifact_size",
    "toc_candidate_sha256",
    "selected_page_index",
    "metrics",
    "title_sequence_digest",
    "expected_toc_applicability",
    "expected_route",
}
_HISTORICAL_METRIC_KEYS = {
    "candidate_present",
    "source_line_count",
    "printed_entry_count",
    "printed_sequence_valid",
    "structural_entry_count",
    "structural_coverage_ratio",
    "structural_sequence_valid",
    "unique_structures",
    "source_order_valid",
    "hierarchy_valid",
    "aggregate_character_count",
    "chunk_count",
}


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
    evidence_mode: str = "direct_candidate_analysis",
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
    except TocTransformationExhausted:
        chunks = []

    return {
        "evidence_mode": evidence_mode,
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
        "candidate_rejection_reason": quality.get("candidate_rejection_reason"),
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
    result = analyze_toc_candidate(
        candidate_text or None,
        toc_page_list=candidate_pages,
        object_sha256_prefix=hashlib.sha256(pdf_bytes).hexdigest()[
            :PREFLIGHT_HASH_PREFIX_LENGTH
        ],
        object_size=len(pdf_bytes),
        page_count=len(page_texts),
        evidence_mode=FRESH_SCAN_EVIDENCE_MODE,
    )
    result["candidate_selection_policy"] = (
        "fresh_pages_with_structural_coverage_at_or_above_threshold"
    )
    result["historical_detector_selection_replayed"] = False
    return result


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain only the sanitized schema keys")


def _selected_route(applicability):
    if applicability in {
        TOC_APPLICABILITY_UNUSABLE_CANDIDATE,
        TOC_APPLICABILITY_NONE,
    }:
        return "process_no_toc"
    if applicability == TOC_APPLICABILITY_PRINTED_NUMBERED:
        return "process_toc_with_page_numbers"
    return "process_toc_no_page_numbers"


def replay_sanitized_historical_records(payload: dict) -> dict:
    """Replay approved detector metrics without PDF bytes or candidate text."""
    _require_exact_keys(payload, _HISTORICAL_ROOT_KEYS, "historical record")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported historical record schema")
    if payload["evidence_mode"] != HISTORICAL_EVIDENCE_MODE:
        raise ValueError("historical record evidence mode is invalid")

    source = payload["source_object"]
    _require_exact_keys(source, _HISTORICAL_SOURCE_KEYS, "source object")
    if (
        not isinstance(source["object_sha256_prefix"], str)
        or not re.fullmatch(
            rf"[0-9a-f]{{{PREFLIGHT_HASH_PREFIX_LENGTH}}}",
            source["object_sha256_prefix"],
        )
        or not isinstance(source["object_size"], int)
        or source["object_size"] < 1
        or not isinstance(source["page_count"], int)
        or source["page_count"] < 1
    ):
        raise ValueError("source object identity is invalid")

    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("historical records require at least one artifact")

    replayed = []
    classifications = set()
    routes = set()
    for artifact in artifacts:
        _require_exact_keys(artifact, _HISTORICAL_ARTIFACT_KEYS, "artifact")
        metrics = artifact["metrics"]
        _require_exact_keys(metrics, _HISTORICAL_METRIC_KEYS, "artifact metrics")
        if not isinstance(artifact["observation_id"], str) or not re.fullmatch(
            r"historical_attempt_[1-9]\d*", artifact["observation_id"]
        ):
            raise ValueError("artifact observation identity is invalid")
        if any(
            not isinstance(artifact[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", artifact[key])
            for key in (
                "artifact_sha256",
                "toc_candidate_sha256",
                "title_sequence_digest",
            )
        ):
            raise ValueError("artifact digest is invalid")
        if (
            not isinstance(artifact["artifact_size"], int)
            or artifact["artifact_size"] < 1
            or not isinstance(artifact["selected_page_index"], int)
            or artifact["selected_page_index"] < 0
            or artifact["selected_page_index"] >= source["page_count"]
        ):
            raise ValueError("artifact bounds are invalid")
        integer_metrics = (
            "source_line_count",
            "printed_entry_count",
            "structural_entry_count",
            "aggregate_character_count",
            "chunk_count",
        )
        boolean_metrics = (
            "candidate_present",
            "printed_sequence_valid",
            "structural_sequence_valid",
            "unique_structures",
            "source_order_valid",
            "hierarchy_valid",
        )
        if any(
            not isinstance(metrics[key], int)
            or isinstance(metrics[key], bool)
            or metrics[key] < 0
            for key in integer_metrics
        ):
            raise ValueError("artifact count metric is invalid")
        if any(not isinstance(metrics[key], bool) for key in boolean_metrics):
            raise ValueError("artifact boolean metric is invalid")
        coverage = metrics["structural_coverage_ratio"]
        if (
            not isinstance(coverage, (int, float))
            or isinstance(coverage, bool)
            or coverage < 0.0
            or coverage > 1.0
        ):
            raise ValueError("artifact coverage metric is invalid")

        applicability = classify_toc_candidate_metrics(metrics)
        route = _selected_route(applicability)
        if applicability != artifact["expected_toc_applicability"]:
            raise ValueError("historical applicability changed")
        if route != artifact["expected_route"]:
            raise ValueError("historical route changed")
        classifications.add(applicability)
        routes.add(route)
        replayed.append(
            {
                "observation_id": artifact["observation_id"],
                "artifact_sha256": artifact["artifact_sha256"],
                "artifact_size": artifact["artifact_size"],
                "toc_candidate_sha256": artifact["toc_candidate_sha256"],
                "selected_page_index": artifact["selected_page_index"],
                "metrics": dict(metrics),
                "title_sequence_digest": artifact["title_sequence_digest"],
                "toc_applicability": applicability,
                "selected_route": route,
            }
        )

    return {
        "evidence_mode": HISTORICAL_EVIDENCE_MODE,
        "source_object": dict(source),
        "artifact_count": len(replayed),
        "stable_identical_classification": len(classifications) == 1,
        "stable_identical_route": len(routes) == 1,
        "artifacts": replayed,
    }


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
    parser.add_argument("--historical-records", type=Path)
    parser.add_argument("--object-sha256-prefix")
    parser.add_argument("--object-size", type=int)
    args = parser.parse_args()
    selected_inputs = sum(
        bool(value)
        for value in (args.pdf, args.artifact_json, args.historical_records)
    )
    if selected_inputs != 1:
        parser.error(
            "provide exactly one PDF, --artifact-json, or --historical-records"
        )

    if args.pdf:
        result = analyze_pdf_bytes(args.pdf.read_bytes())
    elif args.historical_records:
        result = replay_sanitized_historical_records(
            json.loads(args.historical_records.read_bytes())
        )
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
            evidence_mode="historical_raw_artifact_analysis",
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
