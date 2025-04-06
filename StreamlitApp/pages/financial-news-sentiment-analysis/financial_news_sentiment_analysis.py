from typing import Any, Iterator, Mapping
from ollama import chat
from pydantic import BaseModel
import pandas as pd
from gnews import GNews
import streamlit as st
import yfinance as yf


# Set page config
st.set_page_config(page_title="Financial News Sentiment Analysis", page_icon="📈")
st.title(body="Financial News Sentiment Analysis")


# Define Structured Output for financial news sentiment analysis
class FinancialSentimentAnalysis(BaseModel):
    """Structured Output for financial sentiment news analysis."""

    sentiment: str
    future_looking: bool


# Get S&P 500 symbols using yfinance
def get_sp500_symbols() -> dict[Any, Any]:
    """Get S&P 500 symbols using yfinance."""

    sp500: pd.DataFrame = pd.read_html(
        io="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return dict(zip(sp500["Security"], sp500["Symbol"]))


# Financial news sentiment analysis using llm
def analyze_stock_news(symbol) -> pd.DataFrame:
    """Financial news sentiment analysis using llm."""

    # Fetch news articles
    google_news: GNews = GNews()
    news: Any | list[dict[Any, Any]] | list[Any] | None = google_news.get_news(
        key=symbol
    )

    # Check if news is None or empty
    if not news:
        st.error(body=f"No news found for {symbol}. Please try another company.")
        return pd.DataFrame(columns=["title", "sentiment", "future_looking"])

    # Extract top 10 news titles
    news_titles: list[Any] = [article["title"] for article in news[:10]]

    # Initialize response store result
    results: list[dict[str, Any]] = []

    # Create llm to analyze news articles
    # model: str = "deepseek-r1:14b"
    # model: str = "llama3.2:3b"
    # model: str = "openthinker:7b"
    model: str = "gemma3:4b"

    for title in news_titles:
        response: Mapping[Any, Any] | Iterator[Mapping[Any, Any]] = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a financial analyst expert. When analyzing headlines:
                    - For future_looking, mark True if the headline:
                    * Makes predictions about future performance
                    * Discusses upcoming events or releases
                    * Contains words like 'will', 'expected', 'forecast', 'outlook', 'guidance'
                    * Mentions price targets or future valuations
                    * Discusses future market trends or opportunities
                    - For sentiment, analyze based on:
                    * positive: bullish indicators, growth, achievements, upgrades
                    * negative: bearish indicators, declines, risks, downgrades
                    * neutral: factual reporting without clear positive/negative bias""",
                },
                {
                    "role": "user",
                    "content": f"""Analyze this financial headline: "{title}"

                    Consider:
                    1. Sentiment: Is it positive, negative, or neutral?
                    2. Future-looking: Does it contain predictions, forecasts, or forward-looking statements about the stock's performance, company plans, or market outlook?

                    Respond in JSON format matching the FinancialSentimentAnalysis schema.""",
                },
            ],
            format="json",
        )

        # Get the last message from the response
        if isinstance(response, Iterator):
            response_content = ""
            for chunk in response:
                if isinstance(chunk, dict) and "message" in chunk:
                    response_content = chunk["message"]["content"]
        else:
            response_content = response["message"]["content"]

        # Debug the response content
        st.write(f"Raw response for '{title}': {response_content}")

        # Clean up the response content if needed
        # Sometimes the model might return text before or after the JSON
        import json
        import re

        # Try to extract JSON from the response using regex
        json_match = re.search(r"\{[\s\S]*\}", response_content)
        if json_match:
            try:
                # Validate that it's proper JSON by parsing it
                extracted_json = json_match.group(0)
                json.loads(extracted_json)  # Just to validate
                response_content = extracted_json
            except json.JSONDecodeError:
                # If it's not valid JSON, we'll use the original response
                pass

        try:
            # Parse the response into financial sentiment analysis model
            financial_sentiment_analysis: FinancialSentimentAnalysis = (
                FinancialSentimentAnalysis.model_validate_json(
                    json_data=response_content
                )
            )
        except Exception as e:
            st.error(body=f"Error parsing response for '{title}': {str(e)}")
            # Provide a default fallback value
            financial_sentiment_analysis = FinancialSentimentAnalysis(
                sentiment="neutral", future_looking=False
            )
            st.warning(body=f"Using fallback values for '{title}'")

        # Store the results
        results.append(
            {
                "title": title,
                "sentiment": financial_sentiment_analysis.sentiment,
                "future_looking": financial_sentiment_analysis.future_looking,
            }
        )

    # Converts the results to DataFrame
    df: pd.DataFrame = pd.DataFrame(data=results)
    return df


def main():

    # Get stock symbols and create dropdown
    stocks = get_sp500_symbols()
    selected_company = st.selectbox(
        "Select a company",
        options=list(stocks.keys()),
        format_func=lambda x: f"{x} ({stocks[x]})",
    )

    if selected_company:
        symbol = stocks[selected_company]
        if st.button(f"Analyze {symbol} News"):
            with st.spinner(f"Analyzing news for {selected_company}..."):
                df = analyze_stock_news(symbol)

                # Display results
                st.subheader("Sentiment Analysis Results")
                st.dataframe(df)

                # Display summary statistics
                positive_count = len(df[df["sentiment"] == "positive"])
                negative_count = len(df[df["sentiment"] == "negative"])
                neutral_count = len(df[df["sentiment"] == "neutral"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Positive", positive_count)
                col2.metric("Negative", negative_count)
                col3.metric("Neutral", neutral_count)


if __name__ == "__main__":
    main()
