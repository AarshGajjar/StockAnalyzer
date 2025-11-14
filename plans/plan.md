## Blueprint for Stock Analysis Chatbot POC

### **Architecture Overview**

```
User Input → Streamlit UI → Python Backend → Data Layer → LLM Analysis → Response
```

---

### **1. Frontend (Streamlit)**

**Components needed:**
- **Stock input field** - Text input for symbol (e.g., "RELIANCE", "TCS")
- **Chat interface** - `st.chat_message()` and `st.chat_input()` for conversation
- **Session state management** - Store conversation history and fetched stock data
- **Loading indicators** - Show when fetching data or analyzing

**Flow:**
1. User enters stock symbol first time → Fetch all data → Cache in session
2. User asks questions → Pass cached data + question to LLM
3. Display responses in chat format

---

### **2. Data Layer**

**Data sources (pick 2-3 for POC):**

**Option A: APIs (Easiest)**
- **Yahoo Finance (yfinance library)** - Free, has Indian stocks, covers basics
- **NSE India website** - Has unofficial endpoints (corporate info, announcements)
- **Screener.in** - Can scrape public pages

**Option B: Web Scraping**
- **Screener.in** - Company page has financials, ratios, peer comparison
- **Moneycontrol** - News, corporate actions, quarterly results
- **NSE announcements** - Corporate actions, shareholding patterns

**Data structure to collect:**
```python
stock_data = {
    'basic_info': {
        'name': str,
        'sector': str,
        'market_cap': float,
        'current_price': float
    },
    'financials': {
        'revenue': [],  # last 5 years
        'profit': [],
        'eps': [],
        'roe': float,
        'debt_to_equity': float
    },
    'ratios': {
        'pe': float,
        'pb': float,
        'dividend_yield': float
    },
    'news': [
        {'title': str, 'date': str, 'summary': str}
    ],
    'peer_comparison': {}
}
```

---

### **3. Data Fetching Module**

**Structure:**
```python
# data_fetcher.py

class StockDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = {}
    
    def fetch_basic_info(self):
        # Use yfinance or scrape
        pass
    
    def fetch_financials(self):
        # Scrape from Screener.in or use API
        pass
    
    def fetch_news(self):
        # Scrape Moneycontrol recent news
        pass
    
    def get_all_data(self):
        # Orchestrate all fetches
        # Return combined dictionary
        pass
```

**Caching strategy:**
- Store in Streamlit session state for current session
- Optional: Save to Supabase with timestamp for 24hr cache

---

### **4. LLM Integration**

**Two approaches:**

**Option A: Gemini/OpenAI/Anthropic API**
- Send stock data as context + user question
- Best for nuanced analysis

**Option B: Local LLM (OLLAMA)**
- Similar approach
- Zero cost

**Prompt structure:**
```python
system_prompt = """You are a stock analysis assistant specializing in 
Indian equities. You provide fundamental analysis based on financial data.

Always cite specific metrics when making claims. Be objective and mention 
both positives and negatives."""

def analyze_stock(stock_data, user_question):
    context = f"""
    Stock Data for {stock_data['basic_info']['name']}:
    
    Financials: {json.dumps(stock_data['financials'])}
    Ratios: {json.dumps(stock_data['ratios'])}
    Recent News: {json.dumps(stock_data['news'][:3])}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\nQuestion: {user_question}"}
    ]
    
    # Call API
    # Return response
```

---

### **5. Conversation Flow**

**Initial interaction:**
```
User: "TCS"
Bot: "Fetching data for TCS..." → Shows company overview
Bot: "What would you like to know about TCS?"
```

**Follow-up questions:**
```
User: "Is it a good value buy?"
Bot: Analyzes P/E, P/B vs industry, growth rates → Provides opinion

User: "Compare with Infosys"
Bot: Fetches Infosys data → Comparative analysis
```

**State management:**
```python
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []
```

---

### **6. File Structure**

```
stock-chatbot/
├── app.py                 # Streamlit main file
├── data_fetcher.py        # Data scraping/API calls
├── llm_analyzer.py        # LLM integration
├── utils.py               # Helper functions
├── requirements.txt
└── .env                   # API keys
```

---

### **7. MVP Feature Set**

**Must have:**
- Single stock symbol input
- Fetch 3 data types: financials, ratios, recent news
- Chat interface with conversation history
- Basic analysis (valuation, growth, risk)

**Nice to have:**
- Compare 2 stocks
- Custom analysis criteria input
- Export analysis as PDF

**Skip for POC:**
- User authentication
- Supabase caching (use session state only)
- Sentiment analysis (just show raw news)
- Historical data beyond 5 years

---


### **9. Key Technical Decisions**

**For scraping:**
- Use **BeautifulSoup** or **Playwright** (if JS-heavy sites)
- Add delays (2-3 sec) to avoid blocks
- Handle errors gracefully (show partial data)

**For LLM:**
- Stream responses for better UX
- Keep context window under 30k tokens (truncate old data)
- Add retry logic for API failures

**For caching:**
- Session state for POC
- Add Supabase later if users request history

---