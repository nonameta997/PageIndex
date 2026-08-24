import os
import json
import copy
import math
import random
import re
from .utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed


TOC_CONTINUATION_MAX_ATTEMPTS = 5
LARGE_NUMBERED_TOC_MIN_ENTRIES = 96
NUMBERED_TOC_CHUNK_MAX_ENTRIES = 48
NUMBERED_TOC_CHUNK_MAX_CHARS = 12000
NUMBERED_TOC_AGGREGATE_MAX_ENTRIES = 512
NUMBERED_TOC_AGGREGATE_MAX_CHARS = 120000
TOC_OFFSET_MIN_INDEPENDENT_MATCHES = 3
TOC_OFFSET_MIN_AGREEMENT_RATIO = 0.75
TOC_OFFSET_MAX_RESIDUAL = 1
STRUCTURAL_TOC_MIN_ENTRIES = 24
STRUCTURAL_TOC_MIN_COVERAGE_RATIO = 0.90
PAGE_LOCATION_MAX_STRUCTURE_ENTRIES = 48
PAGE_LOCATION_MAX_STRUCTURE_CHARS = 12000
PAGE_LOCATION_MAX_PAGE_GROUP_TOKENS = 12000
PAGE_LOCATION_MAX_PAGE_GROUPS = 16
PAGE_LOCATION_MAX_OUTPUT_TOKENS = 6000
PAGE_LOCATION_MAX_ATTEMPTS = 3
NO_TOC_MAX_TITLE_CHARS = 160
NO_TOC_MAX_TITLE_WORDS = 24
NO_TOC_MIN_DOCUMENT_COVERAGE_RATIO = 0.50
GENERATED_TOC_CORRECTION_MAX_ATTEMPTS = 3
TOC_APPLICABILITY_PRINTED_NUMBERED = "printed_page_numbered"
TOC_APPLICABILITY_STRUCTURAL_NUMBERED = "structurally_numbered_without_page_column"
TOC_APPLICABILITY_UNUSABLE_CANDIDATE = "unusable_detector_candidate"
TOC_APPLICABILITY_NONE = "no_toc"
_NUMBERED_TOC_START = re.compile(
    r"^\s*(?P<structure>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<body>\S.*)$"
)
_NUMBERED_TOC_PAGE = re.compile(
    r"^(?P<title>.+?)\s*(?:\.{2,}|…+|:{1,}|-{2,}|\t+)\s*"
    r"(?P<page>\d+)\s*$"
)


class TocTransformationExhausted(RuntimeError):
    """A deterministic TOC transformation or quality gate was exhausted."""

    def __init__(self, stage, reason):
        self.stage = stage
        self.reason_code = reason
        super().__init__(f"{stage}: {reason}")


def _structure_key(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise TocTransformationExhausted("toc_validation", "invalid_structure")
    return tuple(int(part) for part in value.split("."))


def _validate_toc_sequence(
    items,
    *,
    page_field="page",
    require_pages=False,
    min_page=None,
    max_page=None,
    validation_state=None,
    return_state=False,
):
    if not isinstance(items, list) or not items:
        raise TocTransformationExhausted("toc_validation", "empty_result")

    state = validation_state or {}
    seen = set(state.get("seen", ()))
    previous_structure = state.get("previous_structure")
    previous_page = state.get("previous_page")
    normalized = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            raise TocTransformationExhausted("toc_validation", "invalid_item")
        structure = str(raw_item.get("structure") or "").strip()
        structure_key = _structure_key(structure)
        if structure in seen:
            raise TocTransformationExhausted("toc_validation", "duplicate_structure")
        if previous_structure is not None and structure_key <= previous_structure:
            raise TocTransformationExhausted("toc_validation", "structure_out_of_order")
        if len(structure_key) > 1:
            parent = ".".join(str(part) for part in structure_key[:-1])
            if parent not in seen:
                raise TocTransformationExhausted("toc_validation", "missing_parent")

        title = str(raw_item.get("title") or "").strip()
        if not title:
            raise TocTransformationExhausted("toc_validation", "empty_title")

        page = raw_item.get(page_field) if page_field is not None else None
        if isinstance(page, str) and page.isdigit():
            page = int(page)
        if require_pages and (not isinstance(page, int) or isinstance(page, bool)):
            raise TocTransformationExhausted("toc_validation", "missing_page")
        if isinstance(page, int) and not isinstance(page, bool):
            if min_page is not None and page < min_page:
                raise TocTransformationExhausted("toc_validation", "page_below_bounds")
            if max_page is not None and page > max_page:
                raise TocTransformationExhausted("toc_validation", "page_above_bounds")
            if previous_page is not None and page < previous_page:
                raise TocTransformationExhausted("toc_validation", "page_out_of_order")
            previous_page = page
        elif page is not None:
            raise TocTransformationExhausted("toc_validation", "invalid_page")

        item = copy.deepcopy(raw_item)
        item["structure"] = structure
        item["title"] = title
        if page_field is not None:
            item[page_field] = page
        normalized.append(item)
        seen.add(structure)
        previous_structure = structure_key
    next_state = {
        "seen": seen,
        "previous_structure": previous_structure,
        "previous_page": previous_page,
    }
    if return_state:
        return normalized, next_state
    return normalized


def _has_numbered_toc_structure(toc_content):
    return any(
        _NUMBERED_TOC_START.match(line.strip())
        for line in str(toc_content or "").splitlines()
        if line.strip()
    )


def _structural_toc_entries(toc_content):
    lines = [
        line.strip()
        for line in str(toc_content or "").splitlines()
        if line.strip()
    ]
    blocks = []
    current = None
    covered_lines = 0
    for line in lines:
        match = _NUMBERED_TOC_START.match(line)
        if match:
            if current is not None:
                blocks.append(current)
            current = [match.group("structure"), match.group("body")]
            covered_lines += 1
        elif current is not None:
            current[1] += " " + line
            covered_lines += 1
    if current is not None:
        blocks.append(current)

    entries = []
    for structure, body in blocks:
        printed_page = _NUMBERED_TOC_PAGE.match(body)
        title = printed_page.group("title") if printed_page else body
        title = " ".join(title.split())
        if title:
            entries.append({"structure": structure, "title": title})
    coverage_ratio = covered_lines / len(lines) if lines else 0.0
    return entries, {
        "source_line_count": len(lines),
        "covered_line_count": covered_lines,
        "coverage_ratio": coverage_ratio,
    }


def _numbered_toc_entries(toc_content):
    blocks = []
    current = None
    for raw_line in str(toc_content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _NUMBERED_TOC_START.match(line)
        if match:
            if current is not None:
                blocks.append(current)
            current = [match.group("structure"), match.group("body")]
        elif current is not None:
            current[1] += " " + line
    if current is not None:
        blocks.append(current)

    entries = []
    for structure, body in blocks:
        page_match = _NUMBERED_TOC_PAGE.match(body)
        if not page_match:
            return []
        entries.append(
            {
                "structure": structure,
                "title": page_match.group("title").strip(),
                "page": int(page_match.group("page")),
            }
        )
    return entries


def _candidate_structure_metrics(entries):
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


def classify_toc_candidate_metrics(metrics):
    """Classify sanitized candidate metrics without requiring source text."""
    if not metrics.get("candidate_present"):
        return TOC_APPLICABILITY_NONE

    resource_stage = metrics.get("resource_exhaustion_stage")
    resource_reason = metrics.get("resource_exhaustion_reason")
    if resource_stage or resource_reason:
        raise TocTransformationExhausted(
            resource_stage or "toc_chunk",
            resource_reason or "resource_limit",
        )

    if (
        metrics.get("printed_entry_count", 0) > 0
        and metrics.get("printed_sequence_valid") is True
    ):
        return TOC_APPLICABILITY_PRINTED_NUMBERED

    if (
        metrics.get("structural_entry_count", 0) >= STRUCTURAL_TOC_MIN_ENTRIES
        and metrics.get("structural_coverage_ratio", 0.0)
        >= STRUCTURAL_TOC_MIN_COVERAGE_RATIO
        and metrics.get("unique_structures") is True
        and metrics.get("source_order_valid") is True
        and metrics.get("hierarchy_valid") is True
        and metrics.get("structural_sequence_valid") is True
    ):
        return TOC_APPLICABILITY_STRUCTURAL_NUMBERED

    return TOC_APPLICABILITY_UNUSABLE_CANDIDATE


def classify_toc_candidate(toc_content, page_index_given_in_toc="no"):
    """Classify provider-detected text using deterministic, sanitized gates."""
    content = str(toc_content or "")
    if not content.strip():
        metrics = {
            "candidate_present": False,
            "source_line_count": 0,
            "printed_entry_count": 0,
            "printed_sequence_valid": False,
            "structural_entry_count": 0,
            "structural_coverage_ratio": 0.0,
            "structural_sequence_valid": False,
            "unique_structures": True,
            "source_order_valid": True,
            "hierarchy_valid": True,
            "candidate_rejection_reason": None,
        }
        return classify_toc_candidate_metrics(metrics), metrics

    structural_entries, structural_metrics = _structural_toc_entries(content)
    metrics = {
        "candidate_present": True,
        "source_line_count": structural_metrics["source_line_count"],
        "printed_entry_count": 0,
        "printed_sequence_valid": False,
        "structural_entry_count": len(structural_entries),
        "structural_coverage_ratio": structural_metrics["coverage_ratio"],
        "structural_sequence_valid": False,
        **_candidate_structure_metrics(structural_entries),
        "reported_page_index": page_index_given_in_toc == "yes",
        "candidate_rejection_reason": None,
    }

    printed_entries = _numbered_toc_entries(content)
    metrics["printed_entry_count"] = len(printed_entries)
    if printed_entries:
        try:
            _merge_numbered_toc_chunks(_chunk_numbered_toc(printed_entries))
        except TocTransformationExhausted as exc:
            if exc.stage == "toc_chunk":
                raise
            metrics["candidate_rejection_reason"] = exc.reason_code
        else:
            metrics["printed_sequence_valid"] = True
            return classify_toc_candidate_metrics(metrics), metrics

    if _has_numbered_toc_structure(content):
        try:
            _transform_structural_numbered_toc(content)
        except TocTransformationExhausted as exc:
            if exc.stage == "toc_chunk":
                raise
            metrics["candidate_rejection_reason"] = exc.reason_code
        else:
            metrics["structural_sequence_valid"] = True
            return classify_toc_candidate_metrics(metrics), metrics

    if metrics["candidate_rejection_reason"] is None:
        metrics["candidate_rejection_reason"] = "no_usable_numbered_structure"
    return classify_toc_candidate_metrics(metrics), metrics


def classify_toc_applicability(toc_content, page_index_given_in_toc="no"):
    applicability, _metrics = classify_toc_candidate(
        toc_content,
        page_index_given_in_toc,
    )
    return applicability


def _chunk_numbered_toc(entries):
    if len(entries) > NUMBERED_TOC_AGGREGATE_MAX_ENTRIES:
        raise TocTransformationExhausted("toc_chunk", "aggregate_entry_limit")
    aggregate_chars = sum(
        len(item["structure"]) + len(item["title"]) + 24 for item in entries
    )
    if aggregate_chars > NUMBERED_TOC_AGGREGATE_MAX_CHARS:
        raise TocTransformationExhausted("toc_chunk", "aggregate_character_limit")

    chunks = []
    chunk = []
    chunk_chars = 0
    for item in entries:
        item_chars = len(item["structure"]) + len(item["title"]) + 24
        if item_chars > NUMBERED_TOC_CHUNK_MAX_CHARS:
            raise TocTransformationExhausted("toc_chunk", "entry_too_large")
        if chunk and (
            len(chunk) >= NUMBERED_TOC_CHUNK_MAX_ENTRIES
            or chunk_chars + item_chars > NUMBERED_TOC_CHUNK_MAX_CHARS
        ):
            chunks.append(chunk)
            chunk = []
            chunk_chars = 0
        chunk.append(copy.deepcopy(item))
        chunk_chars += item_chars
    if chunk:
        chunks.append(chunk)
    return chunks


def _validate_numbered_toc_chunk(chunk, validation_state=None):
    return _validate_toc_sequence(
        chunk,
        require_pages=True,
        min_page=1,
        validation_state=validation_state,
        return_state=True,
    )


def _validate_structural_toc_chunk(chunk, validation_state=None):
    return _validate_toc_sequence(
        chunk,
        page_field=None,
        validation_state=validation_state,
        return_state=True,
    )


def _merge_numbered_toc_chunks(chunks):
    merged = []
    validation_state = None
    for chunk in chunks:
        normalized, validation_state = _validate_numbered_toc_chunk(
            chunk,
            validation_state,
        )
        merged.extend(normalized)
    return merged


def _merge_structural_toc_chunks(chunks):
    merged = []
    validation_state = None
    for chunk in chunks:
        normalized, validation_state = _validate_structural_toc_chunk(
            chunk,
            validation_state,
        )
        merged.extend(normalized)
    return merged


def _transform_large_numbered_toc(toc_content):
    entries = _numbered_toc_entries(toc_content)
    if len(entries) < LARGE_NUMBERED_TOC_MIN_ENTRIES:
        return None
    chunks = _chunk_numbered_toc(entries)
    return _merge_numbered_toc_chunks(chunks)


def _transform_structural_numbered_toc(toc_content):
    entries, metrics = _structural_toc_entries(toc_content)
    if len(entries) < STRUCTURAL_TOC_MIN_ENTRIES:
        raise TocTransformationExhausted(
            "toc_structure", "insufficient_structural_entries"
        )
    if metrics["coverage_ratio"] < STRUCTURAL_TOC_MIN_COVERAGE_RATIO:
        raise TocTransformationExhausted(
            "toc_structure", "structural_coverage_below_threshold"
        )
    chunks = _chunk_numbered_toc(entries)
    return _merge_structural_toc_chunks(chunks), metrics, chunks


def _validate_generated_no_toc_candidate(
    items,
    *,
    start_index,
    page_count,
):
    """Fail closed before generated no-TOC output can become a stored tree."""
    if page_count < 1:
        raise TocTransformationExhausted("no_toc_quality", "empty_document")

    normalized = _validate_toc_sequence(
        convert_physical_index_to_int(items),
        page_field="physical_index",
        require_pages=True,
        min_page=start_index,
        max_page=start_index + page_count - 1,
    )
    triples = set()
    evidence_projections = set()
    sibling_pages = set()
    for item in normalized:
        title = " ".join(item["title"].split())
        words = title.split()
        if len(title) > NO_TOC_MAX_TITLE_CHARS:
            raise TocTransformationExhausted(
                "no_toc_quality", "title_character_limit"
            )
        if len(words) > NO_TOC_MAX_TITLE_WORDS:
            raise TocTransformationExhausted(
                "no_toc_quality", "title_word_limit"
            )
        if "\n" in item["title"] or "\r" in item["title"]:
            raise TocTransformationExhausted(
                "no_toc_quality", "multiline_title"
            )

        normalized_title = title.casefold()
        triple = (
            item["structure"],
            normalized_title,
            item["physical_index"],
        )
        if triple in triples:
            raise TocTransformationExhausted(
                "no_toc_quality", "duplicate_structure_title_page"
            )
        triples.add(triple)

        evidence_projection = (normalized_title, item["physical_index"])
        if evidence_projection in evidence_projections:
            raise TocTransformationExhausted(
                "no_toc_quality", "duplicate_title_page_projection"
            )
        evidence_projections.add(evidence_projection)

        parent = item["structure"].rpartition(".")[0]
        sibling_page = (parent, item["physical_index"])
        if sibling_page in sibling_pages:
            raise TocTransformationExhausted(
                "no_toc_quality", "indistinguishable_same_page_siblings"
            )
        sibling_pages.add(sibling_page)

    minimum_last_page = start_index + math.ceil(
        page_count * NO_TOC_MIN_DOCUMENT_COVERAGE_RATIO
    ) - 1
    if normalized[-1]["physical_index"] < minimum_last_page:
        raise TocTransformationExhausted(
            "no_toc_quality", "insufficient_document_coverage"
        )
    return normalized


def _validate_published_tree_shape(nodes, *, start_index, page_count):
    """Validate the existing stored tree/citation fields without changing shape."""
    maximum_page = start_index + page_count - 1
    seen = set()

    def visit(items):
        if not isinstance(items, list) or not items:
            raise TocTransformationExhausted(
                "tree_publication", "empty_published_tree"
            )
        for node in items:
            if not isinstance(node, dict):
                raise TocTransformationExhausted(
                    "tree_publication", "invalid_published_node"
                )
            title = node.get("title")
            node_start = node.get("start_index")
            node_end = node.get("end_index")
            if not isinstance(title, str) or not title.strip():
                raise TocTransformationExhausted(
                    "tree_publication", "invalid_published_title"
                )
            if (
                not isinstance(node_start, int)
                or isinstance(node_start, bool)
                or not isinstance(node_end, int)
                or isinstance(node_end, bool)
                or node_start < start_index
                or node_end > maximum_page
                or node_start > node_end
            ):
                raise TocTransformationExhausted(
                    "tree_publication", "invalid_published_page_range"
                )
            projection = (" ".join(title.casefold().split()), node_start, node_end)
            if projection in seen:
                raise TocTransformationExhausted(
                    "tree_publication", "duplicate_published_projection"
                )
            seen.add(projection)
            children = node.get("nodes")
            if children is not None:
                visit(children)

    visit(nodes)
    return nodes


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    
    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    Reply format:
    {{
        
        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):    
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")


async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")
    
    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, logger=logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content.get('toc_detected', 'no')


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('completed', 'no')


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('completed', 'no')

def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response
    
    chat_history = [
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": response},    
    ]
    continue_prompt = "please continue the generation of table of contents, directly output the remaining part of the structure"
    for _attempt in range(TOC_CONTINUATION_MAX_ATTEMPTS):
        new_response, finish_reason = llm_completion(model=model, prompt=continue_prompt, chat_history=chat_history, return_finish_reason=True)
        response = response + new_response
        chat_history.append({"role": "user", "content": continue_prompt})
        chat_history.append({"role": "assistant", "content": new_response})
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
        if if_complete == "yes" and finish_reason == "finished":
            break
    else:
        raise TocTransformationExhausted("toc_extraction", "continuation_exhausted")
    
    return response

def detect_page_index(toc_content, model=None):
    print('start detect_page_index')
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content.get('page_index_given_in_toc', 'no')

def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)
    
    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }




def toc_index_extractor(toc, content, model=None):
    print('start toc_index_extractor')
    toc_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content



def toc_transformer(toc_content, model=None, trusted_page_index=False):
    print('start toc_transformer')
    deterministic = (
        _transform_large_numbered_toc(toc_content)
        if trusted_page_index
        else None
    )
    if deterministic is not None:
        return deterministic
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = extract_json(last_complete)
        cleaned_response = convert_page_to_int(last_complete.get('table_of_contents', []))
        return _validate_toc_sequence(cleaned_response)
    
    last_complete = get_json_content(last_complete)
    chat_history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": last_complete},
    ]
    continue_prompt = "Please continue the table of contents JSON structure from where you left off. Directly output only the remaining part."
    position = last_complete.rfind('}')
    if position != -1:
        last_complete = last_complete[:position+2]

    for _attempt in range(TOC_CONTINUATION_MAX_ATTEMPTS):
        new_complete, finish_reason = llm_completion(
            model=model,
            prompt=continue_prompt,
            chat_history=chat_history,
            return_finish_reason=True,
        )

        if new_complete.startswith('```json'):
            new_complete = get_json_content(new_complete)
        last_complete = last_complete + new_complete
        chat_history.append({"role": "user", "content": continue_prompt})
        chat_history.append({"role": "assistant", "content": new_complete})

        if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
        if if_complete == "yes" and finish_reason == "finished":
            break
    else:
        raise TocTransformationExhausted("toc_transformer", "continuation_exhausted")

    last_complete = extract_json(last_complete)
    cleaned_response = convert_page_to_int(last_complete.get('table_of_contents', []))
    return _validate_toc_sequence(cleaned_response)
    



def find_toc_pages(start_page_index, page_list, opt, logger=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index
    
    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        detected_result = toc_detector_single_page(page_list[i][0],model=opt.model)
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1
    
    if not toc_page_list and logger:
        logger.info('No toc found')
        
    return toc_page_list

def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)  
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data


def _validate_printed_mapping_output(
    value,
    expected_items,
    *,
    min_page,
    max_page,
):
    """Validate the printed-TOC locator response before downstream dictionary use."""
    if not isinstance(value, list):
        raise TocTransformationExhausted(
            "toc_page_alignment", "provider_output_shape_invalid"
        )
    if not value:
        raise TocTransformationExhausted(
            "toc_page_alignment", "provider_output_empty"
        )

    expected = {
        (item["structure"], item["title"])
        for item in expected_items
    }
    seen = set()
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_output_item_invalid"
            )

        structure = item.get("structure")
        title = item.get("title")
        if not isinstance(structure, str) or not structure.strip():
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_output_structure_invalid"
            )
        if not isinstance(title, str) or not title.strip():
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_output_title_invalid"
            )
        key = (structure.strip(), title.strip())
        if key not in expected:
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_changed_structure"
            )
        if key in seen:
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_output_duplicate"
            )

        physical_index = item.get("physical_index")
        if isinstance(physical_index, str):
            tag = re.fullmatch(r"<physical_index_(\d+)>", physical_index.strip())
            if tag:
                physical_index = int(tag.group(1))
            elif physical_index.strip().isdigit():
                physical_index = int(physical_index.strip())
        if not isinstance(physical_index, int) or isinstance(physical_index, bool):
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_physical_index_invalid"
            )
        if physical_index < min_page or physical_index > max_page:
            raise TocTransformationExhausted(
                "toc_page_alignment", "provider_physical_index_out_of_bounds"
            )

        normalized.append(
            {
                "structure": key[0],
                "title": key[1],
                "physical_index": physical_index,
            }
        )
        seen.add(key)
    return normalized


def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def page_offset_quality_metrics(pairs):
    candidates_by_title = {}
    for pair in pairs:
        try:
            title = " ".join(str(pair['title']).casefold().split())
            physical_index = int(pair['physical_index'])
            page_number = int(pair['page'])
        except (KeyError, TypeError, ValueError):
            continue
        if not title:
            continue
        candidates_by_title.setdefault(title, set()).add(
            (page_number, physical_index)
        )

    independent = [
        next(iter(matches))
        for matches in candidates_by_title.values()
        if len(matches) == 1
    ]
    differences = [physical - printed for printed, physical in independent]
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    selected_offset = None
    agreement_count = 0
    if difference_counts:
        selected_offset, agreement_count = min(
            difference_counts.items(),
            key=lambda item: (-item[1], abs(item[0]), item[0]),
        )
    match_count = len(independent)
    agreement_ratio = agreement_count / match_count if match_count else 0.0
    max_residual = (
        max(abs(diff - selected_offset) for diff in differences)
        if selected_offset is not None
        else None
    )
    return {
        "raw_match_count": len(pairs),
        "independent_match_count": match_count,
        "selected_offset": selected_offset,
        "agreement_count": agreement_count,
        "agreement_ratio": agreement_ratio,
        "max_residual": max_residual,
    }


def calculate_page_offset(pairs):
    metrics = page_offset_quality_metrics(pairs)
    if metrics["independent_match_count"] < TOC_OFFSET_MIN_INDEPENDENT_MATCHES:
        raise TocTransformationExhausted(
            "toc_page_alignment", "insufficient_independent_matches"
        )
    if metrics["agreement_ratio"] < TOC_OFFSET_MIN_AGREEMENT_RATIO:
        raise TocTransformationExhausted(
            "toc_page_alignment", "offset_agreement_below_threshold"
        )
    if metrics["max_residual"] > TOC_OFFSET_MAX_RESIDUAL:
        raise TocTransformationExhausted(
            "toc_page_alignment", "offset_residual_above_threshold"
        )
    return metrics["selected_offset"]

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    
    return data



def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):    
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]
    
    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))
    
    print('divide page_list to groups', len(subsets))
    return subsets

def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X. 

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format. 
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]    
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    if len(structure) > PAGE_LOCATION_MAX_STRUCTURE_ENTRIES:
        raise TocTransformationExhausted(
            "toc_page_mapping", "structure_chunk_entry_limit"
        )
    structure_chars = sum(
        len(item.get("structure", "")) + len(item.get("title", "")) + 24
        for item in structure
    )
    if structure_chars > PAGE_LOCATION_MAX_STRUCTURE_CHARS:
        raise TocTransformationExhausted(
            "toc_page_mapping", "structure_chunk_character_limit"
        )
    if count_tokens(part, model) > PAGE_LOCATION_MAX_PAGE_GROUP_TOKENS:
        raise TocTransformationExhausted(
            "toc_page_mapping", "page_group_token_limit"
        )

    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    current_json_raw, finish_reason = llm_completion(
        model=model,
        prompt=prompt,
        return_finish_reason=True,
        max_retries=PAGE_LOCATION_MAX_ATTEMPTS,
        max_tokens=PAGE_LOCATION_MAX_OUTPUT_TOKENS,
    )
    if finish_reason != "finished":
        raise TocTransformationExhausted(
            "toc_page_mapping", "provider_output_incomplete"
        )
    json_result = extract_json(current_json_raw)
    if not isinstance(json_result, list) or len(json_result) != len(structure):
        raise TocTransformationExhausted(
            "toc_page_mapping", "provider_output_shape_invalid"
        )

    for expected, item in zip(structure, json_result):
        if not isinstance(item, dict):
            raise TocTransformationExhausted(
                "toc_page_mapping", "provider_output_item_invalid"
            )
        if (
            item.get("structure") != expected.get("structure")
            or " ".join(str(item.get("title") or "").split())
            != expected.get("title")
        ):
            raise TocTransformationExhausted(
                "toc_page_mapping", "provider_changed_structure"
            )
        if 'start' in item:
            del item['start']
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text

### add verify completeness
def generate_toc_continue(toc_content, part, model=None):
    print('start generate_toc_continue')
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \
    
    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]    

    Directly return the additional part of the final JSON structure. Do not output anything else."""
    prompt += (
        "\nUse a natural source heading when present. If no short heading exists, "
        "synthesize a concise multilingual title instead of copying a body sentence. "
        f"Every title must be at most {NO_TOC_MAX_TITLE_WORDS} words and "
        f"{NO_TOC_MAX_TITLE_CHARS} characters."
    )

    prompt = prompt + '\nGiven text\n:' + part + '\nPrevious tree structure\n:' + json.dumps(toc_content, indent=2)
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if finish_reason == 'finished':
        return extract_json(response)
    raise TocTransformationExhausted(
        "no_toc_generation", "provider_output_incomplete"
    )
    
### add verify completeness
def generate_toc_init(part, model=None):
    print('start generate_toc_init')
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. 

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            
        ],


    Directly return the final JSON structure. Do not output anything else."""
    prompt += (
        "\nUse a natural source heading when present. If no short heading exists, "
        "synthesize a concise multilingual title instead of copying a body sentence. "
        f"Every title must be at most {NO_TOC_MAX_TITLE_WORDS} words and "
        f"{NO_TOC_MAX_TITLE_CHARS} characters."
    )

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

    if finish_reason == 'finished':
         return extract_json(response)
    raise TocTransformationExhausted(
        "no_toc_generation", "provider_output_incomplete"
    )

def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number= generate_toc_init(group_texts[0], model)
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)    
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f'generate_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number

def _map_structural_toc_to_pages(
    toc_items,
    page_list,
    *,
    start_index=1,
    model=None,
    logger=None,
):
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    group_texts = page_list_to_group_text(
        page_contents,
        token_lengths,
        max_tokens=PAGE_LOCATION_MAX_PAGE_GROUP_TOKENS,
    )
    if len(group_texts) > PAGE_LOCATION_MAX_PAGE_GROUPS:
        raise TocTransformationExhausted(
            "toc_page_mapping", "page_group_count_limit"
        )
    logger.info(f'len(group_texts): {len(group_texts)}')

    mapped_items = []
    for chunk in _chunk_numbered_toc(toc_items):
        mapped_chunk = copy.deepcopy(chunk)
        for group_text in group_texts:
            mapped_chunk = add_page_number_to_toc(
                group_text,
                mapped_chunk,
                model,
            )
        mapped_items.extend(mapped_chunk)

    mapped_items = convert_physical_index_to_int(mapped_items)
    if any(item.get("physical_index") is None for item in mapped_items):
        raise TocTransformationExhausted(
            "toc_page_mapping", "incomplete_tree_coverage"
        )
    return _validate_toc_sequence(
        mapped_items,
        page_field="physical_index",
        require_pages=True,
        min_page=start_index,
        max_page=start_index + len(page_list) - 1,
    )


def process_toc_no_page_numbers(
    toc_content,
    toc_page_list,
    page_list,
    start_index=1,
    model=None,
    logger=None,
    toc_applicability=TOC_APPLICABILITY_STRUCTURAL_NUMBERED,
):
    if toc_applicability == TOC_APPLICABILITY_STRUCTURAL_NUMBERED:
        toc_items, metrics, chunks = _transform_structural_numbered_toc(
            toc_content
        )
        logger.info({
            "mode": "deterministic_structural_toc",
            "entry_count": len(toc_items),
            "chunk_count": len(chunks),
            "coverage_ratio": metrics["coverage_ratio"],
        })
        return _map_structural_toc_to_pages(
            toc_items,
            page_list,
            start_index=start_index,
            model=model,
            logger=logger,
        )

    raise TocTransformationExhausted(
        "toc_policy", "unsupported_no_page_toc_applicability"
    )



def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None, logger=None):
    toc_with_page_number = toc_transformer(
        toc_content,
        model,
        trusted_page_index=True,
    )
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))
    
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    end_page_index = min(start_page_index + toc_check_page_num, len(page_list))
    for page_index in range(start_page_index, end_page_index):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, model)
    toc_with_physical_index = _validate_printed_mapping_output(
        toc_with_physical_index,
        toc_no_page_number,
        min_page=start_page_index + 1,
        max_page=end_page_index,
    )
    logger.info({
        "mode": "printed_toc_page_alignment",
        "mapped_entry_count": len(toc_with_physical_index),
    })

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    logger.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, model=model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number



##check if needed to process none page numbers
def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break
            
            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_index - start_index
                if list_index >= 0 and list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    
    return toc_items




def check_toc(page_list, opt=None):
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        print('no toc found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    else:
        print('toc found')
        toc_json = toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json['page_index_given_in_toc'] == 'yes':
            print('index found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1
            
            while (toc_json['page_index_given_in_toc'] == 'no' and 
                   current_start_index < len(page_list) and 
                   current_start_index < opt.toc_check_page_num):
                
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )
                
                if len(additional_toc_pages) == 0:
                    break

                additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model)
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    print('index found')
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            print('index not found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}






################### fix incorrect toc #########################################################
async def single_toc_item_index_fixer(section_title, content, model=None):
    toc_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    response = await llm_acompletion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    physical_index = json_content.get('physical_index')
    if physical_index is None:
        return None
    return convert_physical_index_to_int(physical_index)



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, logger=None):
    print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    
    end_index = len(page_list) + start_index - 1
    
    correction_range_count = 0
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        nonlocal correction_range_count
        list_index = incorrect_item['list_index']
        
        # Check if list_index is valid
        if list_index < 0 or list_index >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }
        
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1
        
        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index
        
        correction_range_count += 1

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            # Add bounds checking to prevent IndexError
            page_list_idx = page_index - start_index
            if page_list_idx >= 0 and page_list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_list_idx][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
            else:
                continue
        content_range = ''.join(page_contents)
        
        physical_index_int = await single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        
        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print("Generated TOC item correction failed")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    logger.info({
        "mode": "toc_item_correction",
        "requested_count": len(incorrect_results),
        "range_count": correction_range_count,
        "completed_count": len(results),
        "invalid_count": len(invalid_results),
    })

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(
    toc_with_page_number,
    page_list,
    incorrect_results,
    start_index=1,
    max_attempts=3,
    model=None,
    logger=None,
    return_attempts=False,
):
    print('start fix_incorrect_toc')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        
        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model, logger)
                
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break
    
    if return_attempts:
        return current_toc, current_incorrect, fix_attempt
    return current_toc, current_incorrect


def _generated_page_order_state(items, *, start_index, page_count):
    """Return a sanitized correction state without changing source ordering."""
    converted = convert_physical_index_to_int(copy.deepcopy(items))
    if not isinstance(converted, list) or not converted:
        raise TocTransformationExhausted(
            "no_toc_correction", "provider_output_malformed"
        )

    normalized = []
    pages = []
    for item in converted:
        if not isinstance(item, dict):
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_malformed"
            )
        structure = item.get("structure")
        title = item.get("title")
        physical_index = item.get("physical_index")
        try:
            _structure_key(structure)
        except TocTransformationExhausted as exc:
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_malformed"
            ) from exc
        if not isinstance(title, str) or not title.strip():
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_malformed"
            )
        if (
            not isinstance(physical_index, int)
            or isinstance(physical_index, bool)
            or physical_index < start_index
            or physical_index > start_index + page_count - 1
        ):
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_malformed"
            )
        normalized_item = copy.deepcopy(item)
        normalized_item["structure"] = structure.strip()
        normalized_item["title"] = title.strip()
        normalized_item["physical_index"] = physical_index
        normalized.append(normalized_item)
        pages.append(physical_index)

    violation_pairs = [
        (index - 1, index)
        for index in range(1, len(pages))
        if pages[index] < pages[index - 1]
    ]
    violation_score = (
        len(violation_pairs),
        sum(pages[left] - pages[right] for left, right in violation_pairs),
    )
    suspect_indices = sorted(
        {index for pair in violation_pairs for index in pair}
    )
    incorrect_results = [
        {
            "list_index": index,
            "title": normalized[index]["title"],
            "physical_index": normalized[index]["physical_index"],
        }
        for index in suspect_indices
    ]
    signature = tuple(
        (
            item["structure"],
            item["title"],
            item["physical_index"],
        )
        for item in normalized
    )
    return normalized, incorrect_results, violation_score, signature


async def _validate_or_correct_generated_no_toc_candidate(
    items,
    page_list,
    *,
    start_index,
    model,
    logger,
    max_attempts=GENERATED_TOC_CORRECTION_MAX_ATTEMPTS,
):
    """Correct only generated page-order failures under one bounded budget."""
    current = copy.deepcopy(items)
    attempts = 0
    while True:
        try:
            validated = _validate_generated_no_toc_candidate(
                current,
                start_index=start_index,
                page_count=len(page_list),
            )
            return validated, attempts
        except TocTransformationExhausted as exc:
            if not (
                exc.stage == "toc_validation"
                and exc.reason_code == "page_out_of_order"
            ):
                if attempts:
                    raise TocTransformationExhausted(
                        "no_toc_correction", "provider_output_invalid"
                    ) from exc
                raise

        if attempts >= max_attempts:
            raise TocTransformationExhausted(
                "no_toc_correction", "page_order_correction_exhausted"
            )

        before, incorrect, violation_score, signature = (
            _generated_page_order_state(
                current,
                start_index=start_index,
                page_count=len(page_list),
            )
        )
        if not incorrect or violation_score[0] < 1:
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_invalid"
            )

        correction_result = await fix_incorrect_toc(
            copy.deepcopy(before),
            page_list,
            incorrect,
            start_index,
            model,
            logger,
        )
        attempts += 1
        if not isinstance(correction_result, tuple) or len(correction_result) != 2:
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_malformed"
            )
        candidate, _remaining = correction_result
        after, _incorrect, after_score, after_signature = (
            _generated_page_order_state(
                candidate,
                start_index=start_index,
                page_count=len(page_list),
            )
        )
        if tuple(item[:2] for item in after_signature) != tuple(
            item[:2] for item in signature
        ):
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_changed_structure"
            )
        if after_score > violation_score:
            raise TocTransformationExhausted(
                "no_toc_correction", "provider_output_worsened"
            )
        logger.info({
            "mode": "generated_toc_page_order_correction",
            "attempt": attempts,
            "requested_count": len(incorrect),
            "remaining_violation_count": after_score[0],
            "remaining_violation_depth": after_score[1],
            "changed": after_signature != signature,
        })
        current = after




################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    print('start verify_toc')
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break
    
    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []
    
    # Determine which items to check
    if N is None:
        print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_result
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)
    
    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)
    
    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(
    page_list,
    mode=None,
    toc_content=None,
    toc_page_list=None,
    start_index=1,
    opt=None,
    logger=None,
    toc_applicability=TOC_APPLICABILITY_NONE,
):
    print(mode)
    print(f'start_index: {start_index}')
    generated_correction_attempts = 0

    if (
        mode == 'process_toc_with_page_numbers'
        and toc_applicability != TOC_APPLICABILITY_PRINTED_NUMBERED
    ):
        raise TocTransformationExhausted(
            "toc_policy", "numbered_mode_requires_numbered_applicability"
        )
    if mode == 'process_toc_no_page_numbers' and toc_applicability not in {
        TOC_APPLICABILITY_STRUCTURAL_NUMBERED,
    }:
        raise TocTransformationExhausted(
            "toc_policy", "toc_mode_requires_explicit_applicability"
        )
    if mode == 'process_no_toc' and toc_applicability not in {
        TOC_APPLICABILITY_UNUSABLE_CANDIDATE,
        TOC_APPLICABILITY_NONE,
    }:
        raise TocTransformationExhausted(
            "toc_policy", "no_toc_requires_generated_applicability"
        )

    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger)
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(
            toc_content,
            toc_page_list,
            page_list,
            start_index=start_index,
            model=opt.model,
            logger=logger,
            toc_applicability=toc_applicability,
        )
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)
            
    if mode == 'process_no_toc':
        (
            toc_with_page_number,
            generated_correction_attempts,
        ) = await _validate_or_correct_generated_no_toc_candidate(
            toc_with_page_number,
            page_list,
            start_index=start_index,
            model=opt.model,
            logger=logger,
        )
    else:
        toc_with_page_number = _validate_toc_sequence(
            convert_physical_index_to_int(toc_with_page_number),
            page_field="physical_index",
            require_pages=True,
            min_page=start_index,
            max_page=start_index + len(page_list) - 1,
        )
    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, 
        len(page_list), 
        start_index=start_index, 
        logger=logger
    )
    
    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)
        
    logger.info({
        'mode': mode,
        'accuracy': accuracy,
        'incorrect_count': len(incorrect_results),
        'generated_correction_attempts': generated_correction_attempts,
    })
    if accuracy == 1.0 and len(incorrect_results) == 0:
        if mode == 'process_no_toc':
            return _validate_generated_no_toc_candidate(
                toc_with_page_number,
                start_index=start_index,
                page_count=len(page_list),
            )
        return _validate_toc_sequence(
            toc_with_page_number,
            page_field="physical_index",
            require_pages=True,
            min_page=start_index,
            max_page=start_index + len(page_list) - 1,
        )
    remaining_correction_attempts = (
        GENERATED_TOC_CORRECTION_MAX_ATTEMPTS - generated_correction_attempts
        if mode == 'process_no_toc'
        else GENERATED_TOC_CORRECTION_MAX_ATTEMPTS
    )
    if (
        accuracy > 0.6
        and len(incorrect_results) > 0
        and remaining_correction_attempts > 0
    ):
        correction_result = await fix_incorrect_toc_with_retries(
            toc_with_page_number,
            page_list,
            incorrect_results,
            start_index=start_index,
            max_attempts=remaining_correction_attempts,
            model=opt.model,
            logger=logger,
            return_attempts=mode == 'process_no_toc',
        )
        if mode == 'process_no_toc':
            (
                toc_with_page_number,
                incorrect_results,
                used_attempts,
            ) = correction_result
            generated_correction_attempts += used_attempts
        else:
            toc_with_page_number, incorrect_results = correction_result
        if mode == 'process_no_toc':
            toc_with_page_number = _validate_generated_no_toc_candidate(
                toc_with_page_number,
                start_index=start_index,
                page_count=len(page_list),
            )
        else:
            toc_with_page_number = _validate_toc_sequence(
                convert_physical_index_to_int(toc_with_page_number),
                page_field="physical_index",
                require_pages=True,
                min_page=start_index,
                max_page=start_index + len(page_list) - 1,
            )
        accuracy, incorrect_results = await verify_toc(
            page_list,
            toc_with_page_number,
            start_index=start_index,
            model=opt.model,
        )
        if accuracy == 1.0 and len(incorrect_results) == 0:
            if mode == 'process_no_toc':
                return _validate_generated_no_toc_candidate(
                    toc_with_page_number,
                    start_index=start_index,
                    page_count=len(page_list),
                )
            return _validate_toc_sequence(
                toc_with_page_number,
                page_field="physical_index",
                require_pages=True,
                min_page=start_index,
                max_page=start_index + len(page_list) - 1,
            )
    raise TocTransformationExhausted("toc_quality", "whole_tree_verification_failed")
        
 
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])
    
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model, logger=logger)
        
        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]
        
        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']
        
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    
    return node

async def tree_parser(page_list, opt, doc=None, logger=None):
    check_toc_result = check_toc(page_list, opt)
    logger.info(check_toc_result)

    toc_applicability, candidate_metrics = classify_toc_candidate(
        check_toc_result.get("toc_content"),
        check_toc_result.get("page_index_given_in_toc", "no"),
    )
    logger.info({
        "toc_applicability": toc_applicability,
        "candidate_metrics": candidate_metrics,
    })
    if toc_applicability == TOC_APPLICABILITY_PRINTED_NUMBERED:
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_toc_with_page_numbers', 
            start_index=1, 
            toc_content=check_toc_result['toc_content'], 
            toc_page_list=check_toc_result['toc_page_list'], 
            opt=opt,
            logger=logger,
            toc_applicability=toc_applicability)
    elif toc_applicability == TOC_APPLICABILITY_STRUCTURAL_NUMBERED:
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_toc_no_page_numbers',
            start_index=1,
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'],
            opt=opt,
            logger=logger,
            toc_applicability=toc_applicability,
        )
    elif toc_applicability == TOC_APPLICABILITY_UNUSABLE_CANDIDATE:
        logger.info({
            'mode': 'process_no_toc',
            'reason': 'explicit_final_degradation_after_candidate_quality_rejection',
            'toc_applicability': toc_applicability,
            'candidate_metrics': candidate_metrics,
        })
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_no_toc',
            start_index=1,
            opt=opt,
            logger=logger,
            toc_applicability=toc_applicability,
        )
    else:
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt,
            logger=logger,
            toc_applicability=TOC_APPLICABILITY_NONE)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model, logger=logger)
    
    # Filter out items with None physical_index before post_processings
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    
    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)
    return _validate_published_tree_shape(
        toc_tree,
        start_index=1,
        page_count=len(page_list),
    )


def page_index_main(doc, opt=None):
    logger = JsonLogger(doc)
    
    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print('Parsing PDF...')
    page_list = get_page_tokens(doc, model=opt.model)

    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum([page[1] for page in page_list])})

    async def page_index_builder():
        structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
        if opt.if_add_node_id == 'yes':
            write_node_id(structure)    
        if opt.if_add_node_text == 'yes':
            add_node_text(structure, page_list)
        if opt.if_add_node_summary == 'yes':
            if opt.if_add_node_text == 'no':
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, model=opt.model)
            if opt.if_add_node_text == 'no':
                remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':
                # Create a clean structure without unnecessary fields for description generation
                clean_structure = create_clean_structure_for_description(structure)
                doc_description = generate_doc_description(clean_structure, model=opt.model)
                structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())


def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    if not toc_with_page_number:
        return toc_with_page_number
    
    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []
    
    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                if logger:
                    logger.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")
    
    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")
        
    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")
     
    return toc_with_page_number
