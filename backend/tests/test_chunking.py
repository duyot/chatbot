from app.services.ingestion import Element


def _el(type_, text, page=1, level=None, bbox=None, id_=None, confidence=None):
    return Element(
        id=id_ or f"e-{text[:8]}", page=page, type=type_, text=text,
        bbox=bbox, level=level, confidence=confidence,
    )


# --- keep_elements (drop rules) --------------------------------------------

def test_keep_elements_drops_page_headers_and_footers():
    from app.services.chunking import keep_elements
    kept = keep_elements([
        _el("page_header", "ACME Confidential"),
        _el("paragraph", "Real content"),
        _el("page_footer", "Page 3 of 12"),
    ])
    assert [e.text for e in kept] == ["Real content"]


def test_keep_elements_drops_uncaptioned_figure():
    from app.services.chunking import keep_elements
    kept = keep_elements([_el("figure", "chart.png"), _el("paragraph", "Body")])
    assert [e.type for e in kept] == ["paragraph"]


def test_keep_elements_keeps_caption_that_follows_a_figure():
    from app.services.chunking import keep_elements
    kept = keep_elements([
        _el("figure", "chart.png"),
        _el("caption", "Figure 1: revenue by region"),
    ])
    assert [e.type for e in kept] == ["caption"]
    assert kept[0].text == "Figure 1: revenue by region"


def test_keep_elements_drops_blank_text():
    from app.services.chunking import keep_elements
    kept = keep_elements([_el("paragraph", "   "), _el("paragraph", "Real")])
    assert [e.text for e in kept] == ["Real"]


def test_keep_elements_respects_configured_drop_list(mocker):
    from app.services import chunking
    mocker.patch.object(chunking.settings, "drop_element_types", ["caption"])
    kept = chunking.keep_elements([
        _el("caption", "dropped now"),
        _el("page_header", "kept now"),
    ])
    assert [e.text for e in kept] == ["kept now"]


# --- split_sections (heading stack) ----------------------------------------

def test_split_sections_builds_nested_heading_path():
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "3. Financials", level=1),
        _el("paragraph", "intro"),
        _el("heading", "3.2 Revenue", level=2),
        _el("paragraph", "revenue prose"),
    ])
    assert [s.heading_path for s in sections] == [
        "3. Financials",
        "3. Financials > 3.2 Revenue",
    ]
    assert [e.text for e in sections[1].elements] == ["revenue prose"]


def test_split_sections_pops_stack_on_shallower_heading():
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "A", level=1),
        _el("heading", "A.1", level=2),
        _el("paragraph", "deep"),
        _el("heading", "B", level=1),
        _el("paragraph", "shallow"),
    ])
    assert sections[-1].heading_path == "B"
    assert [e.text for e in sections[-1].elements] == ["shallow"]


def test_split_sections_tolerates_level_jumps():
    """h1 -> h3 with no h2 must not crash or produce an empty path segment."""
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "A", level=1),
        _el("heading", "A.0.1", level=3),
        _el("paragraph", "body"),
    ])
    assert sections[-1].heading_path == "A > A.0.1"


def test_split_sections_content_before_any_heading_has_empty_path():
    from app.services.chunking import split_sections
    sections = split_sections([_el("paragraph", "preamble")])
    assert sections[0].heading_path == ""
    assert [e.text for e in sections[0].elements] == ["preamble"]


def test_split_sections_omits_sections_with_no_body():
    """A heading immediately followed by another heading yields no section."""
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "A", level=1),
        _el("heading", "A.1", level=2),
        _el("paragraph", "body"),
    ])
    assert len(sections) == 1
    assert sections[0].heading_path == "A > A.1"


def test_split_sections_defaults_missing_level_to_one():
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "No level given", level=None),
        _el("paragraph", "body"),
    ])
    assert sections[0].heading_path == "No level given"


def test_split_sections_empty_input_returns_empty():
    from app.services.chunking import split_sections
    assert split_sections([]) == []


# --- with_heading ----------------------------------------------------------

def test_with_heading_prefixes_path():
    from app.services.chunking import with_heading
    assert with_heading("A > B", "body text") == "A > B\n\nbody text"


def test_with_heading_returns_body_unchanged_when_path_empty():
    from app.services.chunking import with_heading
    assert with_heading("", "body text") == "body text"


# --- pack_prose ------------------------------------------------------------

def test_pack_prose_keeps_small_elements_in_one_group():
    from app.services.chunking import pack_prose
    els = [_el("paragraph", "short one"), _el("paragraph", "short two")]
    groups = pack_prose(els, max_tokens=1500)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_pack_prose_splits_when_budget_exceeded():
    from app.services.chunking import pack_prose
    els = [_el("paragraph", "word " * 200, id_=f"e{i}") for i in range(6)]
    groups = pack_prose(els, max_tokens=300)
    assert len(groups) > 1
    # every element lands in exactly one group, order preserved
    flat = [e.id for g in groups for e in g]
    assert flat == [e.id for e in els]


def test_pack_prose_never_drops_an_oversized_single_element():
    from app.services.chunking import pack_prose
    els = [_el("paragraph", "word " * 2000)]
    groups = pack_prose(els, max_tokens=100)
    assert len(groups) == 1 and len(groups[0]) == 1


def test_pack_prose_empty_input_returns_empty():
    from app.services.chunking import pack_prose
    assert pack_prose([], max_tokens=1500) == []


# --- build_prose_parent ----------------------------------------------------

def test_build_prose_parent_prefixes_heading_on_parent_and_every_child():
    from app.services.chunking import build_prose_parent
    els = [_el("paragraph", "word " * 200), _el("paragraph", "other " * 200)]
    parent, children = build_prose_parent("A > B", els)

    assert parent.content.startswith("A > B\n\n")
    assert len(children) > 1, "expected the 300-token splitter to produce several children"
    for child in children:
        assert child.content.startswith("A > B\n\n"), \
            "every child needs the header, not just the first"


def test_build_prose_parent_spans_pages():
    from app.services.chunking import build_prose_parent
    els = [_el("paragraph", "on one", page=4), _el("paragraph", "on two", page=5)]
    parent, _ = build_prose_parent("", els)
    assert (parent.page_start, parent.page_end) == (4, 5)


def test_build_prose_parent_child_collects_bboxes_of_overlapping_elements():
    from app.services.chunking import build_prose_parent
    a = _el("paragraph", "alpha text here", bbox=[0.0, 0.0, 1.0, 0.1])
    b = _el("paragraph", "beta text here", bbox=[0.0, 0.2, 1.0, 0.3])
    parent, children = build_prose_parent("", [a, b])

    # Short content -> one child spanning both elements, so it carries both rects.
    assert len(children) == 1
    assert children[0].bbox == [[0.0, 0.0, 1.0, 0.1], [0.0, 0.2, 1.0, 0.3]]


def test_build_prose_parent_skips_none_bboxes():
    from app.services.chunking import build_prose_parent
    els = [_el("paragraph", "no box here", bbox=None)]
    _, children = build_prose_parent("", els)
    assert children[0].bbox == []


def test_build_prose_parent_marks_children_as_text_element_type():
    from app.services.chunking import build_prose_parent
    _, children = build_prose_parent("", [_el("paragraph", "prose")])
    assert children[0].element_type == "text"


def test_build_prose_parent_child_inherits_page_and_confidence():
    from app.services.chunking import build_prose_parent
    el = _el("paragraph", "content", page=7, confidence=0.83)
    _, children = build_prose_parent("", [el])
    assert children[0].page == 7
    assert children[0].ocr_confidence == 0.83


def test_build_prose_parent_source_is_ocr_when_any_element_has_confidence():
    from app.services.chunking import build_prose_parent
    parent, _ = build_prose_parent("", [
        _el("paragraph", "native bit"),
        _el("paragraph", "scanned bit", confidence=0.7),
    ])
    assert parent.source == "ocr"


def test_build_prose_parent_source_is_native_without_confidence():
    from app.services.chunking import build_prose_parent
    parent, _ = build_prose_parent("", [_el("paragraph", "native only")])
    assert parent.source == "native"


def test_build_prose_parent_multiple_children_carry_different_pages_and_bboxes():
    """Comprehensive test: multiple elements on different pages, each large enough
    to produce its own child. Different children carry different, correctly-scoped
    bbox and page values tied to their source element."""
    from app.services.chunking import build_prose_parent

    # Three large elements on different pages, each with distinct bboxes
    el1 = _el("paragraph", "word " * 250, page=1, bbox=[0.0, 0.0, 1.0, 0.3])
    el2 = _el("paragraph", "word " * 250, page=2, bbox=[0.0, 0.3, 1.0, 0.6])
    el3 = _el("paragraph", "word " * 250, page=3, bbox=[0.0, 0.6, 1.0, 1.0])

    parent, children = build_prose_parent("", [el1, el2, el3])

    # Should produce multiple children due to 300-token budget
    assert len(children) >= 3, f"expected at least 3 children, got {len(children)}"

    # Each child should be attributed to one of the three pages
    pages = [child.page for child in children]
    assert 1 in pages, "expected at least one child on page 1"
    assert 2 in pages, "expected at least one child on page 2"
    assert 3 in pages, "expected at least one child on page 3"

    # Verify that children on different pages have different bboxes
    # (at least one child from each page should carry its element's bbox)
    all_bboxes = [child.bbox for child in children if child.bbox]
    assert len(all_bboxes) >= 3, "expected children to carry bboxes from their source elements"

    # Spot-check: a child on page 1 should have bbox in the [0.0-0.3] range
    children_on_page_1 = [c for c in children if c.page == 1]
    children_on_page_2 = [c for c in children if c.page == 2]
    children_on_page_3 = [c for c in children if c.page == 3]

    assert len(children_on_page_1) > 0, "expected at least one child on page 1"
    assert len(children_on_page_2) > 0, "expected at least one child on page 2"
    assert len(children_on_page_3) > 0, "expected at least one child on page 3"

    # Verify that at least one child on page 1 has a bbox from el1
    page_1_with_bbox = [c for c in children_on_page_1 if c.bbox]
    assert len(page_1_with_bbox) > 0, "expected children on page 1 to carry bboxes"
    assert [0.0, 0.0, 1.0, 0.3] in page_1_with_bbox[0].bbox, \
        f"expected el1's bbox in page 1 child, got {page_1_with_bbox[0].bbox}"

    # Verify that at least one child on page 2 has a bbox from el2
    page_2_with_bbox = [c for c in children_on_page_2 if c.bbox]
    assert len(page_2_with_bbox) > 0, "expected children on page 2 to carry bboxes"
    assert [0.0, 0.3, 1.0, 0.6] in page_2_with_bbox[0].bbox, \
        f"expected el2's bbox in page 2 child, got {page_2_with_bbox[0].bbox}"

    # Verify that at least one child on page 3 has a bbox from el3
    page_3_with_bbox = [c for c in children_on_page_3 if c.bbox]
    assert len(page_3_with_bbox) > 0, "expected children on page 3 to carry bboxes"
    assert [0.0, 0.6, 1.0, 1.0] in page_3_with_bbox[0].bbox, \
        f"expected el3's bbox in page 3 child, got {page_3_with_bbox[0].bbox}"
