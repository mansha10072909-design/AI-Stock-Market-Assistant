
import streamlit as st

from graph import graph


# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="AI Stock Market Assistant",
    page_icon="📈",
    layout="wide"
)


# ---------------------------------------
# APPLICATION HEADER
# ---------------------------------------

st.title("📈 AI Stock Market Assistant")

st.caption(
    "LangChain + LangGraph + Financial News RAG"
)

st.info(
    "Educational research application. "
    "Recommendations are simulated, not financial advice."
)


# ---------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------

st.sidebar.header("Stock Research")

ticker = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="RELIANCE.NS"
).strip().upper()

period = st.sidebar.selectbox(
    "Historical Period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

query = st.sidebar.text_area(
    "Research Question",
    value="What are the latest business developments and risks?"
)

run_analysis = st.sidebar.button(
    "Run Stock Analysis",
    type="primary",
    use_container_width=True
)


# ---------------------------------------
# RUN LANGGRAPH WORKFLOW
# ---------------------------------------

if run_analysis:

    if not ticker:
        st.error("Please enter a stock ticker.")

    elif not query.strip():
        st.error("Please enter a research question.")

    else:

        with st.spinner(
            "Our AI research agents are working..."
        ):

            try:

                result = graph.invoke({
                    "ticker": ticker,
                    "period": period,
                    "query": query
                })

                st.session_state["analysis_result"] = result

            except Exception as e:

                st.error(f"Analysis failed: {e}")


# ---------------------------------------
# DISPLAY RESULTS
# ---------------------------------------

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    st.divider()

    st.subheader(
        f"Analysis Results: {ticker}"
    )

    # Recommendation
    recommendation = result.get(
        "recommendation",
        "Unavailable"
    )

    st.subheader("Simulated Recommendation")

    if recommendation == "BUY":
        st.success("BUY")

    elif recommendation == "SELL":
        st.error("SELL")

    else:
        st.warning("HOLD")

    # Market statistics
    market_data = result.get("market_data", {})

    if market_data:

        st.subheader("Market Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Latest Price",
            f"{market_data.get('latest_price', 'N/A')}"
        )

        col2.metric(
            "Period Change",
            f"{market_data.get('percentage_change', 'N/A')}%"
        )

        col3.metric(
            "Highest Price",
            market_data.get("highest_price", "N/A")
        )

        col4.metric(
            "Lowest Price",
            market_data.get("lowest_price", "N/A")
        )

    else:
        st.warning("Market data unavailable.")

    # Market analysis
    with st.expander("Market Analysis", expanded=True):

        st.write(
            result.get(
                "market_analysis",
                "Market analysis unavailable."
            )
        )

    # News analysis
    with st.expander("Financial News Analysis"):

        st.write(
            result.get(
                "news_analysis",
                "News analysis unavailable."
            )
        )

    # Retrieved news
    st.subheader("Retrieved Financial News")

    news_items = result.get("raw_news", [])

    if news_items:

        for item in news_items:

            st.markdown(
                f"**{item.get('title', 'Untitled')}**"
            )

            st.caption(
                f"Publisher: {item.get('publisher', 'Unknown')}"
            )

            if item.get("url"):
                st.markdown(
                    f"[Read original article]({item['url']})"
                )

            st.divider()

    else:

        st.info("No relevant financial news retrieved.")

    # Final report
    st.subheader("Final Research Report")

    st.markdown(
        result.get(
            "final_report",
            "Report unavailable."
        )
    )

    # Workflow status
    st.subheader("Workflow Status")

    st.write(
        "Market data:",
        "Successful" if result.get("data_ok") else "Unavailable"
    )

    st.write(
        "News retrieval:",
        "Successful" if result.get("news_ok") else "Unavailable"
    )


else:

    st.divider()

    st.markdown(
        """
        ### Welcome!

        Enter a stock ticker in the sidebar to begin.

        **Example tickers:**

        - RELIANCE.NS — Reliance Industries
        - TCS.NS — Tata Consultancy Services
        - INFY.NS — Infosys
        - AAPL — Apple Inc.

        Your research report will appear here.
        """
    )


# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "AI Stock Market Assistant | Academic Project | "
    "Simulated recommendations are not financial advice."
)
