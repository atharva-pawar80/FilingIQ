"""
section_split.py
Splits a 10-K's raw HTML into its native "Item" sections (Item 1, 1A, 7, 8...)
BEFORE chunking, so no chunk ever straddles two unrelated sections.

Why this exists: if you skip this and chunk the raw document directly, a
chunk can end up half "Risk Factors" and half "Legal Proceedings" — garbage
context that quietly hurts retrieval precision. Every downstream chunk gets
a clean section label for free, which becomes citation metadata later.
"""

import re
from bs4 import BeautifulSoup

# 10-K Item headers follow a fairly consistent pattern: "Item 1A." etc.
# This regex catches the common variants (with/without period, spacing).
ITEM_HEADER_PATTERN = re.compile(r"item\s+\d+[a-z]?\.?", re.IGNORECASE)

# The sections we actually care about for Q&A — everything else (cover
# page, signatures, exhibits index) gets discarded to keep the index lean.
RELEVANT_SECTIONS = {
    "item 1": "Business",
    "item 1a": "Risk Factors",
    "item 7": "MD&A",
    "item 7a": "Quantitative and Qualitative Market Risk",
    "item 8": "Financial Statements",
}


def html_to_clean_text(html: str) -> str:
    """Strip HTML tags down to readable text, collapsing whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines that HTML->text conversion leaves behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_into_sections(html: str) -> dict:
    """
    Returns {section_label: section_text} for the relevant Items only.
    section_label examples: "Item 1A", "Item 7", "Item 8"
    """
    text = html_to_clean_text(html)
    lines = text.split("\n")

    sections = {}
    current_label = None
    current_lines = []

    for line in lines:
        match = ITEM_HEADER_PATTERN.match(line.strip())
        if match:
            normalized = match.group(0).lower().replace(".", "").strip()
            if normalized in RELEVANT_SECTIONS:
                # Save whatever we were accumulating for the previous section
                if current_label:
                    sections[current_label] = "\n".join(current_lines).strip()
                current_label = normalized
                current_lines = []
                continue

        if current_label:
            current_lines.append(line)

    if current_label:
        sections[current_label] = "\n".join(current_lines).strip()

    # Re-key with human-readable labels for downstream metadata
    return {
        RELEVANT_SECTIONS[key]: value
        for key, value in sections.items()
        if value  # drop empty sections (header matched but no content followed)
    }


if __name__ == "__main__":
    # Local test with a tiny synthetic filing — no network needed.
    sample_html = """
    <html><body>
    <p>Item 1. Business</p>
    <p>We design, manufacture, and sell consumer electronics.</p>
    <p>Item 1A. Risk Factors</p>
    <p>Our business is subject to intense competition and supply chain risk.</p>
    <p>Item 7. Management Discussion and Analysis</p>
    <p>Revenue declined due to weaker demand in the smartphone segment.</p>
    </body></html>
    """
    sections = split_into_sections(sample_html)
    for label, content in sections.items():
        print(f"--- {label} ---")
        print(content)
        print()
