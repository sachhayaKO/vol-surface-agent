"""
SEC EDGAR retrieval for 10-K risk factors and 8-K earnings press releases.

SEC EDGAR doesn't host earnings call transcripts (those aren't SEC
filings) — the closest available "earnings" text is the press release
exhibit (usually EX-99.1) attached to the 8-K filed right after a
quarter closes. This module scopes to what's actually there: 10-K risk
factors and 8-K earnings exhibits.

SEC's fair-access policy requires a descriptive User-Agent identifying
the requester (see https://www.sec.gov/os/webmaster-faq#developers) —
requests without one get rate-limited or blocked.
"""

import re
import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Some filings' primary documents are iXBRL-tagged HTML with XML namespace
# declarations at the top, which makes BeautifulSoup's HTML parser warn
# that it looks like XML. It parses correctly either way (we only need
# get_text()), so this is noise, not a real problem.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SEC_USER_AGENT = "vol-surface-agent research project (sachhaya1@gmail.com)"
HEADERS = {"User-Agent": SEC_USER_AGENT}

_CIK_CACHE: dict[str, str] | None = None


def _load_ticker_to_cik_map() -> dict[str, str]:
    global _CIK_CACHE
    if _CIK_CACHE is not None:
        return _CIK_CACHE

    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=10
    )
    resp.raise_for_status()
    data = resp.json()

    _CIK_CACHE = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }
    return _CIK_CACHE


def get_cik_for_ticker(ticker: str) -> str:
    """Look up a ticker's 10-digit zero-padded CIK. Raises ValueError if
    the ticker isn't in SEC's ticker-to-CIK mapping (e.g. it's an ETF
    with no filer CIK, or an invalid ticker)."""
    mapping = _load_ticker_to_cik_map()
    cik = mapping.get(ticker.upper())
    if cik is None:
        raise ValueError(
            f"{ticker} has no SEC filer CIK (likely an ETF/fund, not an "
            f"individual filer with 10-Ks/8-Ks)"
        )
    return cik


def get_recent_filings(
    cik: str, form_type: str, count: int = 5, item: str | None = None
) -> list[dict]:
    """
    Return the `count` most recent filings of `form_type` (e.g. "10-K",
    "8-K") for a given CIK, each as a dict with accessionNumber,
    filingDate, primaryDocument, and items.

    If `item` is given (e.g. "2.02", the 8-K item code for "Results of
    Operations and Financial Condition"), only filings whose `items`
    field contains it are returned. This is the reliable way to find
    earnings-related 8-Ks — an 8-K can be filed for a dozen unrelated
    reasons (board changes, debt issuance, etc), so filtering on the
    actual disclosed item code beats keyword-matching document text.
    """
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=10
    )
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    matches = []
    for i, form in enumerate(recent["form"]):
        if form != form_type:
            continue
        items = recent["items"][i]
        if item is not None and item not in items.split(","):
            continue
        matches.append(
            {
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
                "items": items,
            }
        )
        if len(matches) >= count:
            break
    return matches


def find_press_release_exhibit(cik: str, accession_number: str) -> str | None:
    """
    Find the earnings press release exhibit filename within an 8-K's
    documents. The submissions API's `primaryDocument` for an earnings
    8-K is typically just the cover page (with inline XBRL tags, no
    real prose) — the actual press release is exhibit 99.1, filed as a
    separate document.

    Filenames for that exhibit aren't consistent across filers (Apple
    uses "a8-kex991....htm", NVIDIA uses "q1fy27pr.htm" — filename
    pattern-matching would need per-filer special-casing and still
    break on the next filer). Instead this parses the human-readable
    filing index page (`{accession}-index.html`), which has a
    structured "Type" column labeling each document (e.g. "EX-99.1")
    independent of how the filer named the file — reliable across
    filers because it's EDGAR's own document classification, not a
    filename guess.

    Returns None if no EX-99.1 is listed, so the caller can fall back
    to primaryDocument.
    """
    accession_no_dashes = accession_number.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/"
        f"{accession_number}-index.html"
    )
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")

    table = soup.find("table", class_="tableFile")
    if table is None:
        return None

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].get_text(strip=True)
        if doc_type.upper() == "EX-99.1":
            link = cells[2].find("a")
            if link is not None:
                return link.get_text(strip=True)
    return None


def fetch_filing_document(cik: str, accession_number: str, primary_document: str) -> str:
    """Fetch a filing's primary document and return its visible text
    (HTML tags stripped)."""
    accession_no_dashes = accession_number.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_no_dashes}/{primary_document}"
    )
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    return soup.get_text(separator="\n")


def extract_risk_factors_section(full_text: str) -> str | None:
    """
    Heuristic extraction of the "Item 1A. Risk Factors" section from a
    10-K's full text: find the last occurrence of an "Item 1A" heading
    (10-Ks often reference it in the table of contents too, so take the
    last match) up to the next "Item 1B" or "Item 2" heading. This is a
    heuristic, not a robust filing parser — it can miss or over/under
    -capture on filings with unusual formatting. Good enough for this
    project's grounding-context use case; a production system would
    want a proper EDGAR full-text-search / section API instead.
    """
    starts = [
        m.start()
        for m in re.finditer(r"item\s+1a\.?\s+risk\s+factors", full_text, re.IGNORECASE)
    ]
    if not starts:
        return None
    start = starts[-1]

    end_match = re.search(
        r"item\s+1b\.?\s+unresolved|item\s+2\.?\s+properties",
        full_text[start:],
        re.IGNORECASE,
    )
    end = start + end_match.start() if end_match else start + 20_000

    section = full_text[start:end].strip()
    return section if len(section) > 200 else None


def fetch_10k_risk_factors(ticker: str) -> str | None:
    """Fetch the most recent 10-K's risk factors section for a ticker.
    Returns None if no 10-K is found or the section can't be located."""
    cik = get_cik_for_ticker(ticker)
    filings = get_recent_filings(cik, "10-K", count=1)
    if not filings:
        return None
    filing = filings[0]
    text = fetch_filing_document(cik, filing["accessionNumber"], filing["primaryDocument"])
    return extract_risk_factors_section(text)


def fetch_latest_earnings_release(ticker: str) -> str | None:
    """
    Fetch the most recent earnings-related 8-K's press release text.
    Identifies earnings 8-Ks via item code 2.02 ("Results of Operations
    and Financial Condition"), then fetches the EX-99.1-style exhibit
    document specifically rather than the primary document (which is
    usually just an XBRL cover page with no real prose — see
    find_press_release_exhibit). Returns None if no earnings 8-K is
    found in the most recent filings.
    """
    cik = get_cik_for_ticker(ticker)
    filings = get_recent_filings(cik, "8-K", count=1, item="2.02")
    if not filings:
        return None
    filing = filings[0]

    exhibit_name = find_press_release_exhibit(cik, filing["accessionNumber"])
    document = exhibit_name or filing["primaryDocument"]
    return fetch_filing_document(cik, filing["accessionNumber"], document)


def chunk_text(text: str, chunk_words: int = 150, overlap_words: int = 30) -> list[str]:
    """
    Split text into overlapping chunks for embedding, sized in words
    (not characters) so chunks stay within the embedding model's token
    limit — all-MiniLM-L6-v2 (the default embedding model here) truncates
    at 256 tokens, and 150 words is comfortably under that even with
    dense filing text. Splits on whitespace boundaries so chunks don't
    cut mid-word.
    """
    words = text.split()
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_words])
        if chunk:
            chunks.append(chunk)
    return chunks
