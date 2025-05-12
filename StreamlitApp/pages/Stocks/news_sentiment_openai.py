from typing import Any, Iterator, Mapping, Dict, cast
from ollama import chat
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd
from gnews import GNews
import streamlit as st


# Define Structured Output for stock news sentiment analysis
class FinancialSentimentAnalysis(BaseModel):
    """Structured Output for stock sentiment news analysis."""

    sentiment: str
    future_looking: bool


# Financial news sentiment analysis using llm
def analyze_stock_news(openai_api_key: str, openai_model: str, symbol: str):
    """Stock news sentiment analysis using llm."""

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

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    model = openai_model

    for title in news_titles:
        response = client.responses.create(
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

        # Store the results as structured data
        results.append(
            {
                "title": title,
                "sentiment": financial_sentiment_analysis.sentiment,
                "future_looking": financial_sentiment_analysis.future_looking,
            }
        )

    # Converts the results to DataFrame
    df_news: pd.DataFrame = pd.DataFrame(data=results)
    index_range: pd.RangeIndex = pd.RangeIndex(start=1, stop=len(df_news) + 1, step=1)
    df_news.index = index_range

    # Display results
    st.subheader(body="AI Stock News Sentiment Analysis Results")
    st.dataframe(data=df_news)

    # Display summary statistics
    positive_count: int = len(df_news[df_news["sentiment"] == "positive"])
    negative_count: int = len(df_news[df_news["sentiment"] == "negative"])
    neutral_count: int = len(df_news[df_news["sentiment"] == "neutral"])

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Positive", value=positive_count)
    col2.metric(label="Negative", value=negative_count)
    col3.metric(label="Neutral", value=neutral_count)
