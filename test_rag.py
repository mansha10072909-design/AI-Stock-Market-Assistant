
from rag import (
    fetch_financial_news,
    create_news_vector_store,
    retrieve_relevant_news
)

ticker = "RELIANCE.NS"

print("\n1. Fetching financial news...")

documents = fetch_financial_news(ticker)

print("Number of news articles:", len(documents))

if documents:

    print("\n2. Creating FAISS vector store...")

    vector_store = create_news_vector_store(documents)

    print("\n3. Searching relevant news...")

    results = retrieve_relevant_news(
        vector_store,
        "What are the latest business developments and risks?",
        k=3
    )

    print("\nRetrieved news:")

    for i, doc in enumerate(results, 1):
        print(f"\nArticle {i}:")
        print("Title:", doc.metadata.get("title"))
        print("Publisher:", doc.metadata.get("publisher"))
        print("Content:", doc.page_content)
        print("URL:", doc.metadata.get("url"))

else:
    print("No financial news was returned. Try again later.")
