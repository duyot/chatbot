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
