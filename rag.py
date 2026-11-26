
import yfinance as yf

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def fetch_financial_news(ticker: str):
    """
    Fetch recent financial news for a stock ticker.
    """

    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news

        documents = []

        for item in news_items:
            content = item.get("content", {})

            title = content.get("title", "")
            summary = content.get("summary", "")

            if not title:
                title = item.get("title", "")

            if not summary:
                summary = title

            if title:
                documents.append(
                    Document(
                        page_content=f"{title}. {summary}",
                        metadata={
                            "ticker": ticker,
                            "title": title,
                            "publisher": content.get("provider", {}).get("displayName", "Unknown"),
                            "url": content.get("canonicalUrl", {}).get("url", ""),
                        }
                    )
                )

        return documents

    except Exception as e:
        print("News retrieval error:", e)
        return []


def create_news_vector_store(documents):
    """
    Convert news documents into embeddings and store them in FAISS.
    """

    if not documents:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def retrieve_relevant_news(vector_store, query: str, k: int = 3):
    """
    Retrieve the most relevant news articles using similarity search.
    """

    if vector_store is None:
        return []

    return vector_store.similarity_search(query, k=k)