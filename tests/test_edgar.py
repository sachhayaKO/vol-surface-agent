"""
Unit tests for the pure text-processing pieces of the EDGAR ingestion
module. No network calls — CIK lookup / filing fetch are integration
paths, exercised manually against live EDGAR (see the notebook), not
unit-tested against a fixture that would just go stale.
"""

from vol_surface_agent.ingestion.edgar import chunk_text, extract_risk_factors_section


def test_chunk_text_respects_word_count_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_words=150, overlap_words=30)

    assert all(len(c.split()) <= 150 for c in chunks)
    # consecutive chunks should share the overlap region
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-30:] == second_words[:30]


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_extract_risk_factors_section_finds_bounded_section():
    text = (
        "Item 1. Business\nsome business text\n"
        "Item 1A. Risk Factors\n" + ("risk content " * 100) + "\n"
        "Item 1B. Unresolved Staff Comments\nnothing to report"
    )
    section = extract_risk_factors_section(text)
    assert section is not None
    assert section.startswith("Item 1A")
    assert "Unresolved Staff Comments" not in section


def test_extract_risk_factors_section_returns_none_when_absent():
    assert extract_risk_factors_section("no matching heading here") is None
