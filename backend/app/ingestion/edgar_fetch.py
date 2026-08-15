"""
edgar_fetch.py
Fetches 10-K filings from SEC EDGAR for a given company.

SEC EDGAR requires a descriptive User-Agent header identifying you
(name/email) — requests without one get blocked. Rate limit: 10 req/sec.
Docs: https://www.sec.gov/os/webmaster-faq#developers
"""

import requests
import time

# CHANGE THIS to your real name/email — SEC requires it and will block
# generic or missing User-Agents.
HEADERS = {"User-Agent": "FilingIQ Student Project you@example.com"}

# A few well-known companies for the demo scope, mapped to their SEC CIK
# (Central Index Key) numbers. CIK is how EDGAR identifies companies —
# not the ticker. You look these up once at https://www.sec.gov/cgi-bin/browse-edgar
COMPANY_CIKS = {
    "AAPL": "0000320193",   # Apple
    "MSFT": "0000789019",   # Microsoft
    "JPM": "0000019617",    # JPMorgan Chase
    "GS": "0000886982",     # Goldman Sachs
    "TSLA": "0001318605",   # Tesla
}


def get_latest_10k_url(ticker: str) -> str:
    """
    Given a ticker in COMPANY_CIKS, find the URL of their most recent
    10-K filing document using EDGAR's submissions JSON API.
    """
    cik = COMPANY_CIKS[ticker]
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    resp = requests.get(submissions_url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    # Find the first filing where form == "10-K"
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            cik_int = int(cik)
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{accession}/{primary_doc}"
            )
            return filing_url

    raise ValueError(f"No 10-K found for {ticker}")


def fetch_filing_html(ticker: str) -> str:
    """Fetch the raw HTML of the latest 10-K for a ticker."""
    url = get_latest_10k_url(ticker)
    time.sleep(0.15)  # stay well under the 10 req/sec rate limit
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.text


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check fetching.
    ticker = "AAPL"
    url = get_latest_10k_url(ticker)
    print(f"Latest 10-K URL for {ticker}: {url}")
    html = fetch_filing_html(ticker)
    print(f"Fetched {len(html)} characters of HTML.")
