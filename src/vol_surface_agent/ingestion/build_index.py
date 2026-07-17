"""
Build (or rebuild) the local Chroma index of earnings/10-K text for the
retriever tool. Run this manually when the corpus needs refreshing —
it's not run automatically on every agent query, since EDGAR fetches are
slow and the underlying filings don't change often.

Usage:
    python -m vol_surface_agent.ingestion.build_index
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from vol_surface_agent.ingestion.edgar import (
    chunk_text,
    fetch_10k_risk_factors,
    fetch_latest_earnings_release,
)

# SPY is deliberately excluded — it's an ETF with no 10-K/earnings
# filings (see docs/ARCHITECTURE.md).
RETRIEVER_TICKERS = ["AAPL", "NVDA"]

PERSIST_DIR = "data/chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_documents(ticker: str) -> list[Document]:
    docs = []

    risk_factors = fetch_10k_risk_factors(ticker)
    if risk_factors:
        for chunk in chunk_text(risk_factors):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"ticker": ticker, "source": "10-K risk factors"},
                )
            )
    else:
        print(f"  WARNING: no 10-K risk factors extracted for {ticker}")

    earnings = fetch_latest_earnings_release(ticker)
    if earnings:
        for chunk in chunk_text(earnings):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"ticker": ticker, "source": "earnings release (8-K EX-99.1)"},
                )
            )
    else:
        print(f"  WARNING: no earnings release found for {ticker}")

    return docs


def main() -> None:
    all_docs = []
    for ticker in RETRIEVER_TICKERS:
        print(f"Fetching filings for {ticker}...")
        docs = build_documents(ticker)
        print(f"  {len(docs)} chunks")
        all_docs.extend(docs)

    print(f"\nEmbedding {len(all_docs)} chunks with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="earnings_10k",
    )
    print(f"Persisted index to {PERSIST_DIR}/")


if __name__ == "__main__":
    main()
