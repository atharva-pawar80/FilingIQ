"""
chunk_tables.py
Handles Item 8 (Financial Statements) — the hard case.

Problem: a raw table like
    Revenue          394,328
    Cost of Sales    223,546
has almost no semantic meaning to an embedding model on its own — "394,328"
means nothing without its row label attached, and a naive text splitter can
slice a table in half.

Solution — dual representation:
  1. Convert each row to a natural-language sentence for EMBEDDING
     ("For fiscal year 2023, Apple's Total Revenue was $394,328 million.")
     This gives the embedding model real semantic content to match against
     a question like "what was Apple's revenue in 2023?"
  2. Keep the original table (as markdown) attached as metadata, so the
     LLM sees the precise structured numbers at GENERATION time — sentence
     form for retrieval, table form for answer accuracy.
"""

import pandas as pd


def table_to_row_sentences(df: pd.DataFrame, company: str, fiscal_year: int, table_name: str) -> list[dict]:
    """
    Converts each row of a financial table DataFrame into one chunk:
    a natural-language sentence (for embedding) + the full table as
    markdown (for the LLM to read at generation time).
    """
    table_markdown = df.to_markdown(index=False)
    chunks = []

    # Assume first column is the row label (e.g. "Total Revenue"), and
    # remaining columns are values per period (e.g. "2023", "2022").
    label_col = df.columns[0]

    for _, row in df.iterrows():
        row_label = str(row[label_col]).strip()
        if not row_label or row_label.lower() == "nan":
            continue

        for period_col in df.columns[1:]:
            value = row[period_col]
            if pd.isna(value):
                continue

            # period_col IS the fiscal year/period (e.g. "FY2023") — don't
            # also hardcode an outer fiscal_year, or you get contradictory
            # sentences like "in fiscal year 2023 ... (FY2022) was X".
            sentence = (
                f"In {company}'s {table_name}, {row_label} for {period_col} "
                f"was {value}."
            )

            chunks.append({
                "text": sentence,
                "metadata": {
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "section": "Financial Statements",
                    "chunk_type": "table",
                    "table_name": table_name,
                    "row_label": row_label,
                    "period": str(period_col),
                    "full_table_markdown": table_markdown,  # for generation-time context
                },
            })

    return chunks


def extract_tables_from_html(html: str) -> list[pd.DataFrame]:
    """
    Extracts all HTML tables from a filing section as DataFrames.
    pandas.read_html works reliably on EDGAR filings since their tables
    are properly tagged <table> elements (not image scans).
    """
    return pd.read_html(html)


if __name__ == "__main__":
    # Local test with a synthetic income statement — no network needed.
    sample_df = pd.DataFrame({
        "Line Item": ["Total Revenue", "Cost of Sales", "Gross Profit"],
        "FY2023": ["394,328", "223,546", "170,782"],
        "FY2022": ["365,817", "212,981", "152,836"],
    })

    result = table_to_row_sentences(
        sample_df, company="Apple", fiscal_year=2023, table_name="Income Statement"
    )
    print(f"Produced {len(result)} chunks from {len(sample_df)} table rows.")
    for c in result[:3]:
        print("-", c["text"])
