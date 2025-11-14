# Stock Analysis Chatbot

An advanced Streamlit-based chatbot for analyzing and comparing Indian stocks using real-time data from multiple sources and AI-powered insights.

## 🚀 Features

-   **📊 Multi-Source Data Fetching**: Gathers data from `yfinance`, NSE India, Screener.in, and Moneycontrol for comprehensive analysis.
-   **🤖 Flexible LLM Integration**: Supports multiple LLM providers for analysis:
    -   OpenAI (GPT models)
    -   Google Gemini
    -   Ollama (for local models like Llama 2)
-   **⚙️ Dynamic Configuration**: Configure LLM providers, API keys, and models directly in the UI. No need to restart the app.
-   **💬 Interactive Chat Interface**: Ask questions about stocks in a conversational manner.
-   **📈 In-Depth Analysis**:
    -   Key financial metrics (P/E, P/B, ROE, Debt-to-Equity, etc.).
    -   Historical price performance charts.
    -   Recent news and corporate announcements.
-   **🔀 Stock Comparison Mode**: Compare 2-3 stocks side-by-side with detailed tables and visualizations.
-   **💾 Smart Caching**: Caches stock data to speed up loading and reduce redundant API calls.
-   **🔄 Force Refresh**: Option to bypass the cache and fetch the latest data.

## 🛠️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd stock-analysis-chatbot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The application uses optional dependencies for some features. For full functionality, you might need to install them.*

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

4.  **Configure LLM Provider (in the app):**
    -   Open the application in your browser.
    -   In the sidebar, select an LLM provider (OpenAI, Gemini, or Ollama).
    -   Enter your API key if required.
    -   The app will fetch available models, which you can select from a dropdown.
    -   If you don't configure an LLM, the app will use a basic rule-based analysis.

## Usage

### Single Stock Analysis

1.  Select the "Single Stock" mode in the sidebar.
2.  Enter a stock symbol (e.g., `TCS`, `RELIANCE`, `INFY`).
3.  Click **"Load Stock Data"**.
4.  View key metrics, performance charts, and other data.
5.  Ask questions in the chat interface, such as:
    -   "Is this a good value buy?"
    -   "What are the key risks for this company?"
    -   "Summarize the recent news."

### Stock Comparison

1.  Select the "Compare Stocks" mode in the sidebar.
2.  Enter 2 or 3 stock symbols in the input fields.
3.  Click **"Compare Stocks"**.
4.  Analyze the comparison table and charts.
5.  Use the chat to ask comparative questions, like:
    -   "Which of these stocks has better profitability?"
    -   "Compare the valuation of these companies."
    -   "Which stock is a better investment for the long term?"

## 📂 Project Structure

```
.
├── app.py              # Main Streamlit application, handles UI and state
├── data_fetcher.py     # Fetches stock data from various sources
├── cache_manager.py    # Manages file-based caching of stock data
├── llm_analyzer.py     # Handles interaction with LLMs for analysis
├── stock_comparison.py # Logic for comparing multiple stocks
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 📝 Notes

-   For Indian stocks listed on the NSE, the app automatically adds the `.NS` suffix for `yfinance`.
-   Stock data is cached in a `.stock_cache` directory. You can use the "🔄" button to force a refresh.
-   If using Ollama, ensure it is installed and running locally. See [ollama.ai](https://ollama.ai) for instructions.
-   The app is designed for informational purposes and is not financial advice.