
import json
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from tools import fetch_stock_data
from rag import (
    fetch_financial_news,
    create_news_vector_store,
    retrieve_relevant_news,
)


# --------------------------------------------------
# 1. SHARED STATE
# --------------------------------------------------

class StockState(TypedDict, total=False):
    ticker: str
    period: str
    query: str

    market_data: dict
    data_ok: bool
    market_analysis: str

    raw_news: list
    news_context: str
    news_ok: bool
    news_analysis: str

    recommendation: str
    final_report: str
    workflow_status: str


# --------------------------------------------------
# 2. MARKET DATA AGENT
# --------------------------------------------------

def market_data_agent(state: StockState):

    ticker = state["ticker"]
    period = state.get("period", "6mo")

    try:
        result = fetch_stock_data.invoke({
            "ticker": ticker,
            "period": period
        })

        market_data = json.loads(result)

        if not market_data.get("success"):
            raise ValueError(market_data.get("message"))

        print("Market Data Agent: Data retrieved successfully.")

        return {
            "market_data": market_data,
            "data_ok": True
        }

    except Exception as e:
        print("Market Data Agent error:", e)

        return {
            "market_data": {},
            "data_ok": False
        }


# --------------------------------------------------
# 3. MARKET ANALYSIS AGENT
# --------------------------------------------------

def market_analysis_agent(state: StockState):

    data = state.get("market_data", {})

    if not data:
        analysis = "Market data is unavailable. No market analysis can be made."
    else:
        change = data.get("percentage_change", 0)

        analysis = f"""
        MARKET ANALYSIS

        Ticker: {state['ticker']}
        Period: {data.get('period')}
        Latest price: {data.get('latest_price')}
        Highest price: {data.get('highest_price')}
        Lowest price: {data.get('lowest_price')}
        Average volume: {data.get('average_volume')}
        Period change: {change}%

        Historical price performance is
        {'positive' if change > 0 else 'negative' if change < 0 else 'flat'}.

        This is a basic statistical summary,
        not an AI-generated financial assessment.
        """

    print("Market Analysis Agent: Analysis completed.")

    return {"market_analysis": analysis}


# --------------------------------------------------
# 4. NEWS RETRIEVAL AGENT
# --------------------------------------------------

def news_retrieval_agent(state: StockState):

    ticker = state["ticker"]
    query = state.get(
        "query",
        "What are the recent business developments and risks?"
    )

    try:
        documents = fetch_financial_news(ticker)

        if not documents:
            print("News Retrieval Agent: No news found.")

            return {
                "raw_news": [],
                "news_context": "No financial news was retrieved.",
                "news_ok": False
            }

        vector_store = create_news_vector_store(documents)

        results = retrieve_relevant_news(
            vector_store,
            query,
            k=3
        )

        raw_news = []

        for doc in results:
            raw_news.append({
                "title": doc.metadata.get("title", "Unknown"),
                "publisher": doc.metadata.get("publisher", "Unknown"),
                "url": doc.metadata.get("url", ""),
                "content": doc.page_content
            })

        news_context = "\n\n".join(
            item["content"] for item in raw_news
        )

        print("News Retrieval Agent: Relevant news retrieved.")

        return {
            "raw_news": raw_news,
            "news_context": news_context,
            "news_ok": bool(raw_news)
        }

    except Exception as e:
        print("News Retrieval Agent error:", e)

        return {
            "raw_news": [],
            "news_context": "News retrieval failed: " + str(e),
            "news_ok": False
        }


# --------------------------------------------------
# 5. NEWS ANALYSIS AGENT
# --------------------------------------------------

def news_analysis_agent(state: StockState):

    news = state.get("raw_news", [])

    if not news:
        analysis = (
            "No relevant financial news was available. "
            "News-based conclusions cannot be drawn."
        )

    else:
        lines = ["RETRIEVED FINANCIAL NEWS"]

        for item in news:
            lines.append(
                f"- {item['title']} "
                f"(Publisher: {item['publisher']})"
            )

        analysis = "\n".join(lines)

        analysis += (
            "\n\nThese are retrieved headlines, not "
            "an AI-generated sentiment assessment."
        )

    print("News Analysis Agent: Analysis completed.")

    return {"news_analysis": analysis}


# --------------------------------------------------
# 6. FINAL DECISION AGENT
# --------------------------------------------------

def final_decision_agent(state: StockState):

    data = state.get("market_data", {})
    market_analysis = state.get(
        "market_analysis",
        "Market analysis unavailable."
    )
    news_analysis = state.get(
        "news_analysis",
        "News analysis unavailable."
    )

    if not data:
        recommendation = "HOLD"
        reason = "Insufficient market data for a meaningful signal."

    else:
        change = data.get("percentage_change", 0)

        if change > 5:
            recommendation = "BUY"
        elif change < -5:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        reason = (
            f"The signal follows a basic historical return rule. "
            f"The observed period return was {change}%."
        )

    report = f"""
    AI STOCK MARKET ASSISTANT
    ========================

    Ticker: {state['ticker']}

    MARKET ANALYSIS
    {market_analysis}

    NEWS ANALYSIS
    {news_analysis}

    SIMULATED RECOMMENDATION: {recommendation}

    JUSTIFICATION
    {reason}

    IMPORTANT:
    This is a basic rule-based educational simulation.
    It is not an AI-generated investment recommendation
    and is not personalized financial advice.
    """

    print("Final Decision Agent: Report generated.")

    return {
        "recommendation": recommendation,
        "final_report": report,
        "workflow_status": "Completed"
    }


# --------------------------------------------------
# 7. CONDITIONAL ROUTING
# --------------------------------------------------

def route_after_market_data(state: StockState):

    if state.get("data_ok"):
        return "market_analysis_agent"

    return "news_retrieval_agent"


# --------------------------------------------------
# 8. BUILD LANGGRAPH WORKFLOW
# --------------------------------------------------

workflow = StateGraph(StockState)

workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("market_analysis_agent", market_analysis_agent)
workflow.add_node("news_retrieval_agent", news_retrieval_agent)
workflow.add_node("news_analysis_agent", news_analysis_agent)
workflow.add_node("final_decision_agent", final_decision_agent)

workflow.add_edge(START, "market_data_agent")

workflow.add_conditional_edges(
    "market_data_agent",
    route_after_market_data,
    {
        "market_analysis_agent": "market_analysis_agent",
        "news_retrieval_agent": "news_retrieval_agent"
    }
)

workflow.add_edge(
    "market_analysis_agent",
    "news_retrieval_agent"
)

workflow.add_edge(
    "news_retrieval_agent",
    "news_analysis_agent"
)

workflow.add_edge(
    "news_analysis_agent",
    "final_decision_agent"
)

workflow.add_edge("final_decision_agent", END)

# Compile the graph
graph = workflow.compile()

print("LangGraph Stock Market Assistant initialized!")
