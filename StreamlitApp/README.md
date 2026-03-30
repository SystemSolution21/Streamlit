# StreamlitApp

A comprehensive Streamlit application featuring multiple interactive dashboards, LLM-powered chatbots, and data visualization tools.

## 📁 Project Structure

```project_structure
StreamlitApp/
├── app.py                          # Main application entry point
├── pages/                          # Multi-page Streamlit apps
│   ├── Stocks/                     # Stock market analysis dashboard
│   │   ├── stocks.py              # Main stocks dashboard
│   │   ├── news_sentiment.py      # Sentiment analysis with Ollama
│   │   ├── news_sentiment_openai.py # Sentiment analysis with OpenAI
│   │   ├── styles/                # CSS styling for stocks dashboard
│   │   └── README.md              # Stocks app documentation
│   ├── llm-examples/              # LLM chatbot examples
│   │   ├── Chatbot.py             # Basic OpenAI chatbot
│   │   ├── pages/
│   │   │   ├── 1_File_Q&A.py      # File-based Q&A with Anthropic
│   │   │   ├── 2_Chat_with_search.py # LangChain search integration
│   │   │   ├── 3_Langchain_Quickstart.py # LangChain basics
│   │   │   ├── 4_Langchain_PromptTemplate.py # Prompt engineering
│   │   │   └── 5_Chat_with_user_feedback.py # Feedback collection
│   │   └── README.md              # LLM examples documentation
│   ├── tutorial/                  # Streamlit tutorials
│   │   └── advanced_dashboard/    # Advanced dashboard tutorial
│   └── financial-news-sentiment-analysis/ # Financial sentiment analysis
├── docs/                          # Documentation
│   ├── README.md                  # Documentation home
│   ├── getting_started.md         # Setup and installation guide
│   ├── streamlit_components.md    # Streamlit components reference
│   ├── streamlit_complete_styling.css # Complete CSS styling guide
│   └── tutorials/                 # Tutorial documentation
├── data/                          # Data files and datasets
├── utils/                         # Utility functions and helpers
├── tests/                         # Test suite
│   ├── test_stocks.py            # Tests for stocks dashboard
│   └── conftest.py               # Pytest configuration
├── examples/                      # Code examples
├── pyproject.toml                # Poetry project configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Features

### 1. Stock Market Dashboard (`pages/Stocks/`)

- **S&P 500 Stock Analysis**: Fetch and visualize stock data from Wikipedia and Stooq
- **Interactive Visualizations**: Plotly charts for price trends, volume analysis, and correlations
- **News Sentiment Analysis**: AI-powered sentiment analysis using:
  - Local Ollama LLM (gemma3:4b)
  - OpenAI GPT models (o4-mini, gpt-4.1-nano, o3-mini, gpt-4o-mini)
- **Data Upload**: Support for CSV/Excel file uploads
- **Advanced Analytics**: Correlation matrices, group analysis, trend analysis

### 2. LLM Examples (`pages/llm-examples/`)

- **Basic Chatbot**: OpenAI-powered conversational interface
- **File Q&A**: Upload documents and ask questions (Anthropic Claude)
- **Chat with Search**: LangChain integration with DuckDuckGo search
- **LangChain Quickstart**: Introduction to LangChain framework
- **Prompt Templates**: Advanced prompt engineering examples
- **User Feedback**: Collect and store user feedback with Trubrics

### 3. Financial News Sentiment Analysis

- Real-time news fetching from Google News
- Structured sentiment analysis output
- Future-looking sentiment indicators

### 4. Tutorial Examples

- Advanced dashboard with filtering and plotting
- Sample data visualization techniques
- Interactive UI components

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- Poetry (recommended) or pip

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd StreamlitApp

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run the main app
streamlit run app.py
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main app
streamlit run app.py
```

## 📦 Dependencies

Key dependencies (from `pyproject.toml`):

- **streamlit** (^1.37.1): Web app framework
- **pandas** (^2.2.2): Data manipulation
- **plotly** (^6.0.0): Interactive visualizations
- **matplotlib** (^3.9.2): Static plotting
- **yfinance** (0.2.40): Stock data fetching
- **openai** (^1.78.0): OpenAI API integration
- **ollama** (^0.1.6): Local LLM integration
- **gnews** (^0.4.1): Google News API
- **pydantic** (^2.11.2): Data validation
- **openpyxl** (^3.1.5): Excel file support

## 🔑 API Keys Required

### OpenAI API Key

1. Go to <https://platform.openai.com/account/api-keys>
2. Click "Create new secret key"
3. Enter the key in the Streamlit sidebar when prompted

### Anthropic API Key (for File Q&A)

1. Get your key from Anthropic
2. Enter in the sidebar of the File Q&A page

### Local LLM (Ollama)

For sentiment analysis without API keys:

```bash
# Install Ollama
# Download from https://ollama.ai

# Pull the model
ollama pull gemma3:4b
```

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_stocks.py

# Run with verbose output
pytest tests/test_stocks.py -v

# Run only non-skipped tests
pytest tests/test_stocks.py -v -k "not skip"
```

Current test coverage:

- ✅ `test_get_sp500_symbols`: Tests S&P 500 symbol fetching
- ✅ `test_fetch_stock_data`: Tests stock data retrieval
- ⏭️ `test_stocks_app_loads`: Integration test (skipped)
- ⏭️ `test_news_sentiment_analysis_with_openai`: UI test (skipped)

## 📖 Usage Examples

### Running the Stocks Dashboard

```bash
streamlit run pages/Stocks/stocks.py
```

Features:

1. Upload CSV/Excel files or fetch live stock data
2. Select from S&P 500 companies
3. Visualize price trends and volume
4. Analyze correlations and statistics
5. Get AI-powered news sentiment analysis

### Running LLM Examples

```bash
# Basic chatbot
streamlit run pages/llm-examples/Chatbot.py

# File Q&A
streamlit run pages/llm-examples/pages/1_File_Q&A.py

# Chat with search
streamlit run pages/llm-examples/pages/2_Chat_with_search.py
```

## 🎨 Customization

### CSS Styling

Custom styles are located in:

- `pages/Stocks/styles/dashboard.css`: Stocks dashboard styling
- `docs/streamlit_complete_styling.css`: Complete styling reference

### Streamlit Components

See `docs/streamlit_components.md` for a comprehensive guide to:

- Text elements (title, header, markdown, etc.)
- Input widgets (text_input, selectbox, etc.)
- Layout elements (columns, sidebar, tabs, etc.)
- Data display (dataframe, charts, metrics, etc.)
- Custom CSS classes and styling

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to <https://share.streamlit.io>
3. Connect your repository
4. Add secrets in the app settings:

   ```toml
   OPENAI_API_KEY='your-key-here'
   ```

### GitHub Codespaces

Click the badge in the LLM examples README to open in Codespaces.

## 📝 Documentation

Detailed documentation is available in the `docs/` directory:

- `getting_started.md`: Setup and installation
- `streamlit_components.md`: Component reference
- `tutorials/`: Step-by-step tutorials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- OpenAI and Anthropic for LLM APIs
- LangChain for agent framework
- Ollama for local LLM support

## 📧 Contact

<https://github.com/SystemSolution21>

---

**Note**: This is a tutorial/learning project demonstrating various Streamlit capabilities including data visualization, LLM integration, and interactive dashboards.
