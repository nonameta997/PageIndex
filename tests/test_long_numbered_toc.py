from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pageindex.page_index import (
    LARGE_NUMBERED_TOC_MIN_ENTRIES,
    NO_TOC_MAX_TITLE_CHARS,
    NO_TOC_MAX_TITLE_WORDS,
    NUMBERED_TOC_AGGREGATE_MAX_CHARS,
    NUMBERED_TOC_AGGREGATE_MAX_ENTRIES,
    NUMBERED_TOC_CHUNK_MAX_ENTRIES,
    STRUCTURAL_TOC_MIN_ENTRIES,
    TOC_APPLICABILITY_NONE,
    TOC_APPLICABILITY_PRINTED_NUMBERED,
    TOC_APPLICABILITY_STRUCTURAL_NUMBERED,
    TOC_APPLICABILITY_UNUSABLE_CANDIDATE,
    TOC_OFFSET_MAX_RESIDUAL,
    TOC_OFFSET_MIN_AGREEMENT_RATIO,
    TOC_OFFSET_MIN_INDEPENDENT_MATCHES,
    TocTransformationExhausted,
    _chunk_numbered_toc,
    _map_structural_toc_to_pages,
    _numbered_toc_entries,
    _transform_structural_numbered_toc,
    _validate_generated_no_toc_candidate,
    _validate_numbered_toc_chunk,
    _validate_toc_sequence,
    calculate_page_offset,
    classify_toc_candidate,
    extract_toc_content,
    fix_incorrect_toc_with_retries,
    meta_processor,
    page_offset_quality_metrics,
    process_toc_no_page_numbers,
    toc_detector_single_page,
    toc_transformer,
    tree_parser,
)
from pageindex.toc_preflight import analyze_toc_candidate


def _long_numbered_toc():
    lines = []
    page = 1
    for root in range(1, 33):
        lines.append(f"{root} Chương {root} ........ {page}")
        lines.append(f"{root}.1 Phạm vi áp dụng {root} ........ {page}")
        page += 1
        title = f"Quy định chi tiết {root}"
        if root == 17:
            lines.append(f"{root}.2 Quy định chi tiết")
            lines.append(f"phần tiếp theo ........ {page}")
        else:
            lines.append(f"{root}.2 {title} ........ {page}")
        page += 1
    return "\n".join(lines)


def _structural_toc(entry_count=STRUCTURAL_TOC_MIN_ENTRIES):
    return "\n".join(
        f"{index} Mục nội dung {index}"
        for index in range(1, entry_count + 1)
    )


def test_long_numbered_toc_is_bounded_and_merged_without_provider_calls():
    raw = _long_numbered_toc()
    parsed = _numbered_toc_entries(raw)
    assert len(parsed) == LARGE_NUMBERED_TOC_MIN_ENTRIES
    chunks = _chunk_numbered_toc(parsed)
    assert len(chunks) == 2
    assert all(len(chunk) <= NUMBERED_TOC_CHUNK_MAX_ENTRIES for chunk in chunks)

    with (
        patch(
            "pageindex.page_index.llm_completion",
            side_effect=AssertionError("large numbered TOC must be deterministic"),
        ),
        patch(
            "pageindex.page_index._validate_numbered_toc_chunk",
            wraps=_validate_numbered_toc_chunk,
        ) as validate_chunk,
    ):
        result = toc_transformer(raw, model="test", trusted_page_index=True)

    assert validate_chunk.call_count == len(chunks)
    assert result == parsed
    assert result[0] == {"structure": "1", "title": "Chương 1", "page": 1}
    assert result[49]["structure"] == "17.1"
    assert result[50] == {
        "structure": "17.2",
        "title": "Quy định chi tiết phần tiếp theo",
        "page": 34,
    }
    assert result[-1]["structure"] == "32.2"
    assert [item["page"] for item in result] == sorted(item["page"] for item in result)


def test_printed_page_parser_requires_explicit_column_evidence():
    assert _numbered_toc_entries("1 Section 1\n2 Another section 2") == []
    assert _numbered_toc_entries(
        "1 Dotted leader .... 1\n2 Tab column\t2\n3 Colon: 3"
    ) == [
        {"structure": "1", "title": "Dotted leader", "page": 1},
        {"structure": "2", "title": "Tab column", "page": 2},
        {"structure": "3", "title": "Colon", "page": 3},
    ]


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        (
            [
                {"structure": str(index + 1), "title": "Title", "page": index + 1}
                for index in range(NUMBERED_TOC_AGGREGATE_MAX_ENTRIES + 1)
            ],
            "aggregate_entry_limit",
        ),
        (
            [
                {
                    "structure": str(index + 1),
                    "title": "x"
                    * (NUMBERED_TOC_AGGREGATE_MAX_CHARS // 100 + 1),
                    "page": index + 1,
                }
                for index in range(100)
            ],
            "aggregate_character_limit",
        ),
    ],
)
def test_numbered_toc_aggregate_limits_fail_typed(entries, reason):
    with pytest.raises(TocTransformationExhausted) as caught:
        _chunk_numbered_toc(entries)
    assert caught.value.stage == "toc_chunk"
    assert caught.value.reason_code == reason


def test_page_offset_quality_requires_independent_consistent_evidence():
    assert TOC_OFFSET_MIN_INDEPENDENT_MATCHES == 3
    assert TOC_OFFSET_MIN_AGREEMENT_RATIO == 0.75
    assert TOC_OFFSET_MAX_RESIDUAL == 1

    with pytest.raises(TocTransformationExhausted) as one_pair:
        calculate_page_offset(
            [{"title": "One", "page": 1, "physical_index": 5}]
        )
    assert one_pair.value.reason_code == "insufficient_independent_matches"

    conflicting = [
        {"title": "One", "page": 1, "physical_index": 5},
        {"title": "Two", "page": 2, "physical_index": 6},
        {"title": "Three", "page": 3, "physical_index": 8},
    ]
    with pytest.raises(TocTransformationExhausted) as conflict:
        calculate_page_offset(conflicting)
    assert conflict.value.reason_code == "offset_agreement_below_threshold"

    consistent = [
        {"title": "One", "page": 1, "physical_index": 5},
        {"title": "Two", "page": 2, "physical_index": 6},
        {"title": "Three", "page": 3, "physical_index": 7},
        {"title": "Four", "page": 4, "physical_index": 9},
    ]
    metrics = page_offset_quality_metrics(consistent)
    assert metrics["agreement_ratio"] == 0.75
    assert metrics["max_residual"] == 1
    assert calculate_page_offset(consistent) == 4


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        (
            [
                {"structure": "1", "title": "One", "physical_index": 1},
                {"structure": "1", "title": "Duplicate", "physical_index": 2},
            ],
            "duplicate_structure",
        ),
        (
            [{"structure": "1.1", "title": "Missing parent", "physical_index": 1}],
            "missing_parent",
        ),
        (
            [
                {"structure": "1", "title": "One", "physical_index": 2},
                {"structure": "2", "title": "Two", "physical_index": 1},
            ],
            "page_out_of_order",
        ),
        (
            [{"structure": "1", "title": "Beyond", "physical_index": 9}],
            "page_above_bounds",
        ),
    ],
)
def test_toc_validation_rejects_duplicate_order_hierarchy_and_page_bounds(items, reason):
    with pytest.raises(TocTransformationExhausted) as caught:
        _validate_toc_sequence(
            items,
            page_field="physical_index",
            require_pages=True,
            min_page=1,
            max_page=4,
        )
    assert caught.value.reason_code == reason


def test_toc_transformer_uses_incremental_history_and_typed_five_pass_exhaustion():
    histories = []

    def completion(*_args, **kwargs):
        histories.append(list(kwargs.get("chat_history") or []))
        return ("{" if not kwargs.get("chat_history") else "x", "max_output_reached")

    with (
        patch("pageindex.page_index.llm_completion", side_effect=completion) as mock_llm,
        patch(
            "pageindex.page_index.check_if_toc_transformation_is_complete",
            return_value="no",
        ),
        pytest.raises(TocTransformationExhausted) as caught,
    ):
        toc_transformer("1 Intro .... 1", model="test")

    assert caught.value.stage == "toc_transformer"
    assert caught.value.reason_code == "continuation_exhausted"
    assert mock_llm.call_count == 6
    assert [len(history) for history in histories] == [0, 2, 4, 6, 8, 10]


def test_malformed_completed_continuation_fails_closed_as_typed_empty_result():
    with (
        patch(
            "pageindex.page_index.llm_completion",
            side_effect=[
                ('{"table_of_contents": [', "max_output_reached"),
                ("not-json", "finished"),
            ],
        ),
        patch(
            "pageindex.page_index.check_if_toc_transformation_is_complete",
            side_effect=["no", "yes"],
        ),
        pytest.raises(TocTransformationExhausted) as caught,
    ):
        toc_transformer("1 Intro .... 1", model="test")

    assert caught.value.stage == "toc_validation"
    assert caught.value.reason_code == "empty_result"


def test_extraction_continuation_backport_grows_history_and_exhausts_typed():
    histories = []

    def completion(*_args, **kwargs):
        histories.append(list(kwargs.get("chat_history") or []))
        return ("partial", "max_output_reached")

    with (
        patch("pageindex.page_index.llm_completion", side_effect=completion) as mock_llm,
        patch(
            "pageindex.page_index.check_if_toc_transformation_is_complete",
            return_value="no",
        ),
        pytest.raises(TocTransformationExhausted) as caught,
    ):
        extract_toc_content("raw", model="test")

    assert caught.value.stage == "toc_extraction"
    assert mock_llm.call_count == 6
    assert [len(history) for history in histories] == [0, 2, 4, 6, 8, 10]


@pytest.mark.parametrize("response", ["", "not json", '{"other": "value"}'])
def test_pr188_safe_detection_defaults_to_no(response):
    with patch("pageindex.page_index.llm_completion", return_value=response):
        assert toc_detector_single_page("content", model="test") == "no"


def test_four_state_candidate_quality_matrix_is_deterministic():
    printed, printed_metrics = classify_toc_candidate(
        "1 Giới thiệu .... 1\n2 Phạm vi .... 3",
        "no",
    )
    structural, structural_metrics = classify_toc_candidate(
        _structural_toc(),
        "no",
    )
    unusable, unusable_metrics = classify_toc_candidate(
        "Lời mở đầu\n1 Nội dung\n1 Trùng lặp\nĐoạn văn tiếp nối",
        "yes",
    )
    no_toc, no_toc_metrics = classify_toc_candidate(None, "no")

    assert printed == TOC_APPLICABILITY_PRINTED_NUMBERED
    assert printed_metrics["printed_entry_count"] == 2
    assert structural == TOC_APPLICABILITY_STRUCTURAL_NUMBERED
    assert structural_metrics["structural_entry_count"] == STRUCTURAL_TOC_MIN_ENTRIES
    assert unusable == TOC_APPLICABILITY_UNUSABLE_CANDIDATE
    assert unusable_metrics == {
        "source_line_count": 4,
        "printed_entry_count": 0,
        "structural_entry_count": 2,
        "structural_coverage_ratio": 0.75,
        "reported_page_index": True,
        "candidate_rejection_reason": "insufficient_structural_entries",
    }
    assert no_toc == TOC_APPLICABILITY_NONE
    assert no_toc_metrics["source_line_count"] == 0


def test_sanitized_preflight_selects_only_quality_gated_routes():
    false_positive = analyze_toc_candidate(
        "Header\n1 First\n1 Duplicate\ncontinuation",
        page_index_given_in_toc="yes",
        toc_page_list=[5],
        object_sha256_prefix="9a20c8e616d444e3",
        object_size=2379606,
        page_count=74,
    )
    printed = analyze_toc_candidate("1 Intro .... 1\n2 End .... 2")
    structured = analyze_toc_candidate(_structural_toc())
    no_toc = analyze_toc_candidate(None)

    assert false_positive["toc_applicability"] == (
        TOC_APPLICABILITY_UNUSABLE_CANDIDATE
    )
    assert false_positive["selected_route"] == "process_no_toc"
    assert false_positive["toc_page_list"] == [5]
    assert false_positive["object_sha256_prefix"] == "9a20c8e616d444e3"
    assert "toc_content" not in false_positive
    assert printed["selected_route"] == "process_toc_with_page_numbers"
    assert structured["selected_route"] == "process_toc_no_page_numbers"
    assert no_toc["selected_route"] == "process_no_toc"


@pytest.mark.asyncio
async def test_unusable_detector_candidate_has_one_explicit_no_toc_degradation():
    calls = []
    logger_messages = []

    async def processor(_pages, mode=None, **_kwargs):
        calls.append(mode)
        return [
            {
                "structure": "1",
                "title": "Generated section",
                "physical_index": 1,
                "appear_start": "yes",
            }
        ]

    async def preserve(items, *_args, **_kwargs):
        return items

    logger = SimpleNamespace(info=logger_messages.append)
    opt = SimpleNamespace(
        model="test", max_page_num_each_node=100, max_token_num_each_node=100000
    )
    with (
        patch(
            "pageindex.page_index.check_toc",
            return_value={
                "toc_content": "Introduction\nMethods",
                "toc_page_list": [0],
                "page_index_given_in_toc": "no",
            },
        ),
        patch("pageindex.page_index.meta_processor", side_effect=processor),
        patch(
            "pageindex.page_index.check_title_appearance_in_start_concurrent",
            side_effect=preserve,
        ),
    ):
        tree = await tree_parser([("body", 1)], opt, logger=logger)

    assert calls == ["process_no_toc"]
    assert tree == [{"title": "Generated section", "start_index": 1, "end_index": 1}]
    assert any(
        message.get("reason")
        == "explicit_final_degradation_after_candidate_quality_rejection"
        for message in logger_messages
        if isinstance(message, dict)
    )


@pytest.mark.asyncio
async def test_structurally_numbered_toc_quality_exhaustion_never_degrades():
    calls = []

    async def processor(_pages, mode=None, **kwargs):
        calls.append((mode, kwargs.get("toc_applicability")))
        raise TocTransformationExhausted(
            "toc_quality", "whole_tree_verification_failed"
        )

    opt = SimpleNamespace(
        model="test", max_page_num_each_node=100, max_token_num_each_node=100000
    )
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with (
        patch(
            "pageindex.page_index.check_toc",
            return_value={
                "toc_content": _structural_toc(),
                "toc_page_list": [0],
                "page_index_given_in_toc": "no",
            },
        ),
        patch("pageindex.page_index.meta_processor", side_effect=processor),
        pytest.raises(TocTransformationExhausted) as caught,
    ):
        await tree_parser([("body", 1)], opt, logger=logger)

    assert caught.value.stage == "toc_quality"
    assert calls == [
        ("process_toc_no_page_numbers", TOC_APPLICABILITY_STRUCTURAL_NUMBERED)
    ]
    assert all(mode != "process_no_toc" for mode, _applicability in calls)


@pytest.mark.asyncio
async def test_genuine_no_toc_uses_only_the_explicit_no_toc_path():
    calls = []

    async def processor(_pages, mode=None, **_kwargs):
        calls.append(mode)
        return [
            {
                "structure": "1",
                "title": "Generated section",
                "physical_index": 1,
                "appear_start": "yes",
            }
        ]

    async def preserve(items, *_args, **_kwargs):
        return items

    opt = SimpleNamespace(
        model="test", max_page_num_each_node=100, max_token_num_each_node=100000
    )
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with (
        patch(
            "pageindex.page_index.check_toc",
            return_value={
                "toc_content": None,
                "toc_page_list": [],
                "page_index_given_in_toc": "no",
            },
        ),
        patch("pageindex.page_index.meta_processor", side_effect=processor),
        patch(
            "pageindex.page_index.check_title_appearance_in_start_concurrent",
            side_effect=preserve,
        ),
    ):
        tree = await tree_parser([("body", 1)], opt, logger=logger)

    assert calls == ["process_no_toc"]
    assert tree == [{"title": "Generated section", "start_index": 1, "end_index": 1}]


def test_structural_candidate_bypasses_monolithic_transformer_and_uses_page_mapper():
    raw = _structural_toc()
    expected, metrics, chunks = _transform_structural_numbered_toc(raw)
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with (
        patch(
            "pageindex.page_index.toc_transformer",
            side_effect=AssertionError("structural candidates bypass monolithic JSON"),
        ),
        patch(
            "pageindex.page_index._map_structural_toc_to_pages",
            return_value=[{"structure": "1", "title": "Mapped", "physical_index": 1}],
        ) as mapper,
    ):
        result = process_toc_no_page_numbers(
            raw,
            [0],
            [("body", 1)],
            model="test",
            logger=logger,
            toc_applicability=TOC_APPLICABILITY_STRUCTURAL_NUMBERED,
        )

    assert metrics["coverage_ratio"] == 1.0
    assert len(chunks) == 1
    assert mapper.call_args.args[0] == expected
    assert result[0]["physical_index"] == 1


def test_page_mapping_is_provider_bounded_and_preserves_structure_and_titles():
    items = [
        {"structure": "1", "title": "Giới thiệu"},
        {"structure": "2", "title": "实施细则"},
    ]
    response = (
        '[{"structure":"1","title":"Giới thiệu","start":"yes",'
        '"physical_index":"<physical_index_1>"},'
        '{"structure":"2","title":"实施细则","start":"yes",'
        '"physical_index":"<physical_index_3>"}]'
    )
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with (
        patch("pageindex.page_index.count_tokens", return_value=1),
        patch(
            "pageindex.page_index.llm_completion",
            return_value=(response, "finished"),
        ) as completion,
    ):
        mapped = _map_structural_toc_to_pages(
            items,
            [("one", 1), ("two", 1), ("three", 1)],
            model="test",
            logger=logger,
        )

    assert [(item["structure"], item["title"]) for item in mapped] == [
        ("1", "Giới thiệu"),
        ("2", "实施细则"),
    ]
    assert [item["physical_index"] for item in mapped] == [1, 3]
    assert completion.call_args.kwargs["max_retries"] == 3
    assert completion.call_args.kwargs["max_tokens"] == 6000


@pytest.mark.parametrize(
    ("response", "finish_reason", "reason"),
    [
        ("[]", "max_output_reached", "provider_output_incomplete"),
        ('{"not":"a-list"}', "finished", "provider_output_shape_invalid"),
        (
            '[{"structure":"2","title":"Changed",'
            '"physical_index":"<physical_index_1>"}]',
            "finished",
            "provider_changed_structure",
        ),
    ],
)
def test_page_mapping_malformed_or_truncated_provider_output_fails_typed(
    response,
    finish_reason,
    reason,
):
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with (
        patch("pageindex.page_index.count_tokens", return_value=1),
        patch(
            "pageindex.page_index.llm_completion",
            return_value=(response, finish_reason),
        ),
        pytest.raises(TocTransformationExhausted) as caught,
    ):
        _map_structural_toc_to_pages(
            [{"structure": "1", "title": "Expected"}],
            [("body", 1)],
            model="test",
            logger=logger,
        )
    assert caught.value.stage == "toc_page_mapping"
    assert caught.value.reason_code == reason


def test_valid_multilingual_generated_tree_satisfies_publication_quality():
    result = _validate_generated_no_toc_candidate(
        [
            {"structure": "1", "title": "Giới thiệu", "physical_index": 1},
            {"structure": "2", "title": "Phạm vi áp dụng", "physical_index": 3},
            {"structure": "3", "title": "实施细则", "physical_index": 6},
        ],
        start_index=1,
        page_count=10,
    )
    assert [item["title"] for item in result] == [
        "Giới thiệu",
        "Phạm vi áp dụng",
        "实施细则",
    ]


@pytest.mark.parametrize(
    ("items", "page_count", "reason"),
    [
        ([], 4, "empty_result"),
        (
            [{"structure": "1", "title": "", "physical_index": 1}],
            4,
            "empty_title",
        ),
        (
            [
                    {
                        "structure": "1",
                        "title": " ".join(
                            "w" for _index in range(NO_TOC_MAX_TITLE_WORDS + 1)
                        ),
                    "physical_index": 3,
                }
            ],
            4,
            "title_word_limit",
        ),
        (
            [
                {
                    "structure": "1",
                    "title": "字" * (NO_TOC_MAX_TITLE_CHARS + 1),
                    "physical_index": 3,
                }
            ],
            4,
            "title_character_limit",
        ),
        (
            [
                {"structure": "1", "title": "One", "physical_index": 1},
                {"structure": "2", "title": "Two", "physical_index": 1},
            ],
            2,
            "indistinguishable_same_page_siblings",
        ),
        (
            [
                {"structure": "1", "title": "Repeated", "physical_index": 1},
                {"structure": "1.1", "title": "Repeated", "physical_index": 1},
            ],
            2,
            "duplicate_title_page_projection",
        ),
        (
            [
                {"structure": "1", "title": "One", "physical_index": 1},
                {"structure": "2", "title": "Two", "physical_index": 3},
            ],
            10,
            "insufficient_document_coverage",
        ),
    ],
)
def test_generated_no_toc_publication_quality_rejects_pathologies(
    items,
    page_count,
    reason,
):
    with pytest.raises(TocTransformationExhausted) as caught:
        _validate_generated_no_toc_candidate(
            items,
            start_index=1,
            page_count=page_count,
        )
    assert caught.value.reason_code == reason


@pytest.mark.asyncio
async def test_generated_tree_verify_failure_is_typed_terminal():
    generated = [
        {"structure": "1", "title": "Giới thiệu", "physical_index": 1},
        {"structure": "2", "title": "Kết luận", "physical_index": 4},
    ]
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    opt = SimpleNamespace(model="test", toc_check_page_num=4)
    with (
        patch("pageindex.page_index.process_no_toc", return_value=generated),
        patch("pageindex.page_index.verify_toc", return_value=(0.5, [{"x": 1}])),
        pytest.raises(TocTransformationExhausted) as caught,
    ):
        await meta_processor(
            [("body", 1)] * 4,
            mode="process_no_toc",
            start_index=1,
            opt=opt,
            logger=logger,
            toc_applicability=TOC_APPLICABILITY_NONE,
        )
    assert caught.value.stage == "toc_quality"
    assert caught.value.reason_code == "whole_tree_verification_failed"


@pytest.mark.asyncio
async def test_generated_tree_correction_attempts_are_bounded_to_three():
    items = [{"structure": "1", "title": "Section", "physical_index": 3}]
    incorrect = [{"list_index": 0, "title": "Section", "physical_index": 3}]
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with patch(
        "pageindex.page_index.fix_incorrect_toc",
        return_value=(items, incorrect),
    ) as fixer:
        result, remaining = await fix_incorrect_toc_with_retries(
            items,
            [("body", 1)] * 4,
            incorrect,
            max_attempts=3,
            model="test",
            logger=logger,
        )

    assert fixer.call_count == 3
    assert result == items
    assert remaining == incorrect


@pytest.mark.asyncio
async def test_usable_printed_candidate_exhaustion_never_degrades():
    calls = []

    async def processor(_pages, mode=None, **kwargs):
        calls.append((mode, kwargs.get("toc_applicability")))
        raise TocTransformationExhausted("toc_quality", "whole_tree_verification_failed")

    opt = SimpleNamespace(
        model="test", max_page_num_each_node=100, max_token_num_each_node=100000
    )
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    with (
        patch(
            "pageindex.page_index.check_toc",
            return_value={
                "toc_content": "1 Intro .... 1\n2 Finish .... 2",
                "toc_page_list": [0],
                "page_index_given_in_toc": "yes",
            },
        ),
        patch("pageindex.page_index.meta_processor", side_effect=processor),
        pytest.raises(TocTransformationExhausted),
    ):
        await tree_parser([("body", 1), ("body", 1)], opt, logger=logger)

    assert calls == [
        ("process_toc_with_page_numbers", TOC_APPLICABILITY_PRINTED_NUMBERED)
    ]
