from typing import Any, Iterator, Mapping, Dict, cast
from ollama import chat
from pydantic import BaseModel
import pandas as pd
from gnews import GNews
import streamlit as st


# Set page config
st.set_page_config(page_title="Financial News Sentiment Analysis", page_icon="📈")
st.title(body="Financial News Sentiment Analysis")


# Define Structured Output for financial news sentiment analysis
class FinancialSentimentAnalysis(BaseModel):
    """Structured Output for financial sentiment news analysis."""

    sentiment: str
    future_looking: bool


# Get S&P 500 symbols
def get_sp500_symbols() -> dict[Any, Any]:
    """Get S&P 500 symbols."""

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
        response: Mapping[str, Any] | Iterator[Mapping[str, Any]] = chat(
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
            format=FinancialSentimentAnalysis.model_json_schema(),  # type: ignore
        )

        # Parse the response into financial sentiment analysis model
        # Cast response to Dict to satisfy type checker
        response_dict: Dict[str, Any] = cast(Dict[str, Any], response)
        financial_sentiment_analysis: FinancialSentimentAnalysis = (
            FinancialSentimentAnalysis.model_validate_json(
                json_data=response_dict["message"]["content"]
            )
        )

        # # For debugging, display the response content
        # st.write(f"{title}: {response}")

        # Store the results as structured data
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


def main() -> None:

    # Get stock symbols and create dropdown
    stocks: Dict[Any, Any] = get_sp500_symbols()
    selected_company: Any | None = st.selectbox(
        label="Select a company",
        options=list(stocks.keys()),
        format_func=lambda x: f"{x} ({stocks[x]})",
    )

    if selected_company:
        symbol: str = stocks[selected_company]
        if st.button(label=f"Analyze {symbol} News"):
            with st.spinner(text=f"Analyzing news for {selected_company}..."):
                df: pd.DataFrame = analyze_stock_news(symbol=symbol)

                # Display results
                st.subheader(body="Sentiment Analysis Results")
                st.dataframe(data=df)

                # Display summary statistics
                positive_count: int = len(df[df["sentiment"] == "positive"])
                negative_count: int = len(df[df["sentiment"] == "negative"])
                neutral_count: int = len(df[df["sentiment"] == "neutral"])

                col1, col2, col3 = st.columns(3)
                col1.metric(label="Positive", value=positive_count)
                col2.metric(label="Negative", value=negative_count)
                col3.metric(label="Neutral", value=neutral_count)


if __name__ == "__main__":
    main()
