"""Tests for shared text utilities."""

from utils.text_utils import strip_markdown


def test_strip_markdown_removes_headings():
    text = "## Title\nSome content\n---\nMore text"
    result = strip_markdown(text)
    assert "Some content" in result
    assert "More text" in result


def test_strip_markdown_preserves_table_content():
    text = "| Col1 | Col2 |\n|---|---|\n| A | B |"
    result = strip_markdown(text)
    assert "Col1" in result or "Col2" in result
    assert "---|---" not in result


def test_strip_markdown_handles_empty_lines():
    text = "Line 1\n\n\nLine 2"
    result = strip_markdown(text)
    assert "Line 1" in result
    assert "Line 2" in result
    # Empty lines should be collapsed
    assert result.count("\n") == 1


def test_strip_markdown_empty_input():
    assert strip_markdown("") == ""


def test_strip_markdown_plain_text_unchanged():
    text = "Just some plain text without markdown"
    assert strip_markdown(text) == text
