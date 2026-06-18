"""Shared text processing utilities."""

import re


def strip_markdown(text: str) -> str:
    """Strip markdown formatting, preserving table structure as plain text."""
    lines = text.split("\n")
    cleaned = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            if not re.match(r"^\|[\s\-:|]+\|$", stripped):
                cleaned.append(stripped.strip("|").replace("|", "").strip())
                in_table = True
            continue
        if in_table and not stripped.startswith("|"):
            in_table = False
        cleaned.append(stripped)
    return "\n".join(cleaned)
