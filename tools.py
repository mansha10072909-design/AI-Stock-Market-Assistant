
import json
import yfinance as yf
from langchain_core.tools import tool


@tool
def fetch_stock_data(ticker: str, period: str = "6mo") -> str:
    """
    Fetch historical stock market data for a given ticker.

    Args:
        ticker: Stock ticker symbol, e.g. RELIANCE.NS or AAPL.
        period: Historical period, e.g. 1mo, 3mo, 6mo, 1y, 2y.
    """

    try:
        stock = yf.Ticker(ticker)

        # Fetch historical stock prices
        history = stock.history(period=period)

        if history.empty:
            return json.dumps({
                "success": False,
                "message": f"No stock data found for {ticker}"
            })

        # Calculate stock statistics
        first_price = float(history["Close"].iloc[0])
        latest_price = float(history["Close"].iloc[-1])

        percentage_change = (
            (latest_price - first_price) / first_price
        ) * 100

        result = {
            "success": True,
            "ticker": ticker,
            "period": period,
            "latest_price": round(latest_price, 2),
            "highest_price": round(float(history["High"].max()), 2),
            "lowest_price": round(float(history["Low"].min()), 2),
            "average_volume": int(history["Volume"].mean()),
            "percentage_change": round(percentage_change, 2),
            "number_of_trading_days": len(history)
        }

        return json.dumps(result, indent=4)

    except Exception as e:
        return json.dumps({
            "success": False,
            "message": str(e)
        })
