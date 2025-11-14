import streamlit as st
from data_fetcher import StockDataFetcher
from cache_manager import StockDataCache
from stock_comparison import compare_stocks, get_comparison_summary
from llm_analyzer import (
    analyze_stock, get_provider_status, _get_provider, 
    OPENAI_AVAILABLE, GEMINI_AVAILABLE, OLLAMA_AVAILABLE,
    get_openai_models, get_gemini_models, get_ollama_models,
    test_provider_connection
)
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config with custom CSS
st.set_page_config(
    page_title="Stock Analysis Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    .comparison-table {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None
if 'comparison_symbols' not in st.session_state:
    st.session_state.comparison_symbols = []
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = None
if 'openai_key' not in st.session_state:
    st.session_state.openai_key = ""
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = ""
if 'openai_model' not in st.session_state:
    st.session_state.openai_model = "gpt-3.5-turbo"
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = "gemini-1.5-flash"
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = "llama2"
if 'ollama_host' not in st.session_state:
    st.session_state.ollama_host = ""
if 'openai_models' not in st.session_state:
    st.session_state.openai_models = []
if 'gemini_models' not in st.session_state:
    st.session_state.gemini_models = []
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []
if 'cache_manager' not in st.session_state:
    st.session_state.cache_manager = StockDataCache(cache_duration_hours=1)
if 'enable_streaming' not in st.session_state:
    st.session_state.enable_streaming = True

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color: white;">📈 Stock Analysis Chatbot</h1>
    <p style="margin:0.5rem 0 0 0; color: white; opacity: 0.9;">AI-Powered Stock Analysis for Indian Markets</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["Single Stock", "Compare Stocks"],
        key="mode_select"
    )
    
    st.divider()
    
    if mode == "Single Stock":
        st.subheader("📊 Stock Selection")
        symbol_input = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., TCS, RELIANCE, INFY",
            key="symbol_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            load_button = st.button("Load Stock Data", type="primary", use_container_width=True)
        with col2:
            refresh_button = st.button("🔄", help="Force refresh", use_container_width=True)
        
        if load_button or refresh_button:
            if symbol_input:
                force_refresh = refresh_button
                cache_status = "🔄 Fetching fresh data..." if force_refresh else "📦 Checking cache..."
                
                with st.spinner(cache_status):
                    try:
                        fetcher = StockDataFetcher(symbol_input, use_cache=True, cache_duration_hours=1)
                        data = fetcher.get_all_data(force_refresh=force_refresh)
                        
                        if data.get('basic_info'):
                            st.session_state.stock_data = data
                            st.session_state.current_symbol = symbol_input.upper()
                            st.session_state.comparison_data = None
                            st.session_state.comparison_symbols = []
                            
                            # Add welcome message
                            data_sources = data.get('data_sources', [])
                            sources_text = ', '.join(data_sources) if data_sources else 'yfinance'
                            
                            from_cache = data.get('_from_cache', False)
                            cache_indicator = "📦 (cached)" if from_cache else "🔄 (fresh)"
                            
                            welcome_msg = f"✅ **{data['basic_info'].get('name', symbol_input)}** {cache_indicator}\n\n"
                            welcome_msg += f"**Sector:** {data['basic_info'].get('sector', 'N/A')}  \n"
                            welcome_msg += f"**Price:** ₹{data['basic_info'].get('current_price', 0):.2f}  \n"
                            welcome_msg += f"**Market Cap:** ₹{data['basic_info'].get('market_cap', 0):,.0f}  \n"
                            
                            price_history = data.get('price_history', {})
                            if price_history.get('1d_change'):
                                change = price_history.get('1d_change', 0)
                                emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                                welcome_msg += f"**1D:** {emoji} {change:.2f}%  \n"
                            
                            welcome_msg += f"\n**Data Sources:** {sources_text}  \n"
                            welcome_msg += "\n💬 Ask me anything about this stock!"
                            
                            st.session_state.messages = [{
                                "role": "assistant",
                                "content": welcome_msg
                            }]
                            st.rerun()
                        else:
                            st.error(f"Could not fetch data for {symbol_input}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a stock symbol")
        
        # Current stock info
        if st.session_state.current_symbol:
            st.divider()
            st.info(f"**Current:** {st.session_state.current_symbol}")
            
            cache_info = st.session_state.cache_manager.get_cache_info(st.session_state.current_symbol)
            if cache_info:
                st.caption(f"💾 Cache: {cache_info['age_hours']:.1f}h old")
            
            if st.button("Clear & Start New", use_container_width=True):
                st.session_state.stock_data = {}
                st.session_state.current_symbol = None
                st.session_state.messages = []
                st.rerun()
    
    else:  # Compare Stocks mode
        st.subheader("🔀 Stock Comparison")
        st.caption("Compare up to 3 stocks side-by-side")
        
        symbol1 = st.text_input("Stock 1", placeholder="TCS", key="comp_symbol1")
        symbol2 = st.text_input("Stock 2", placeholder="INFY", key="comp_symbol2")
        symbol3 = st.text_input("Stock 3 (optional)", placeholder="WIPRO", key="comp_symbol3")
        
        if st.button("Compare Stocks", type="primary", use_container_width=True):
            symbols = [s.strip().upper() for s in [symbol1, symbol2, symbol3] if s.strip()]
            if len(symbols) >= 2:
                with st.spinner("Fetching comparison data..."):
                    try:
                        comparison_data = compare_stocks(symbols, force_refresh=False)
                        st.session_state.comparison_data = comparison_data
                        st.session_state.comparison_symbols = symbols
                        st.session_state.current_symbol = None
                        st.session_state.stock_data = {}
                        st.session_state.messages = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter at least 2 stock symbols")
        
        if st.session_state.comparison_symbols:
            st.divider()
            st.info(f"**Comparing:** {', '.join(st.session_state.comparison_symbols)}")
            if st.button("Clear Comparison", use_container_width=True):
                st.session_state.comparison_data = None
                st.session_state.comparison_symbols = []
                st.rerun()
    
    # LLM Configuration
    st.divider()
    st.subheader("🤖 LLM Settings")
    
    # Streaming toggle
    st.session_state.enable_streaming = st.toggle("Enable Streaming", value=True, help="Stream responses for better UX")
    
    available_providers = []
    if OPENAI_AVAILABLE:
        available_providers.append("OpenAI")
    if GEMINI_AVAILABLE:
        available_providers.append("Gemini")
    if OLLAMA_AVAILABLE:
        available_providers.append("Ollama")
    
    if not available_providers:
        st.error("No LLM libraries installed")
    else:
        provider_options = ["None"] + available_providers
        current_index = 0
        if st.session_state.llm_provider:
            try:
                provider_capitalized = st.session_state.llm_provider.capitalize()
                if provider_capitalized in provider_options:
                    current_index = provider_options.index(provider_capitalized)
            except:
                current_index = 0
        
        selected_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=current_index,
            key="provider_select"
        )
        
        if selected_provider == "None":
            st.session_state.llm_provider = None
        else:
            st.session_state.llm_provider = selected_provider.lower()
        
        provider_config = {}
        
        if st.session_state.llm_provider == "openai":
            new_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_key,
                type="password",
                key="openai_key_input"
            )
            if new_key != st.session_state.openai_key:
                st.session_state.openai_models = []
            st.session_state.openai_key = new_key
            
            if st.session_state.openai_key and not st.session_state.openai_models:
                with st.spinner("Fetching models..."):
                    try:
                        st.session_state.openai_models = get_openai_models(st.session_state.openai_key)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.openai_models = []
            
            if st.session_state.openai_models:
                col1, col2 = st.columns([4, 1])
                with col1:
                    current_model_index = 0
                    if st.session_state.openai_model in st.session_state.openai_models:
                        current_model_index = st.session_state.openai_models.index(st.session_state.openai_model)
                    st.session_state.openai_model = st.selectbox(
                        "Model",
                        options=st.session_state.openai_models,
                        index=current_model_index,
                        key="openai_model_select"
                    )
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("🔄", key="refresh_openai", help="Refresh"):
                        st.session_state.openai_models = []
                        st.rerun()
            else:
                st.session_state.openai_model = st.text_input(
                    "Model",
                    value=st.session_state.openai_model,
                    key="openai_model_input"
                )
            
            if st.session_state.openai_key:
                provider_config = {
                    'provider': 'openai',
                    'openai_key': st.session_state.openai_key,
                    'openai_model': st.session_state.openai_model
                }
        
        elif st.session_state.llm_provider == "gemini":
            new_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.gemini_key,
                type="password",
                key="gemini_key_input"
            )
            if new_key != st.session_state.gemini_key:
                st.session_state.gemini_models = []
            st.session_state.gemini_key = new_key
            
            if st.session_state.gemini_key and not st.session_state.gemini_models:
                with st.spinner("Fetching models..."):
                    try:
                        st.session_state.gemini_models = get_gemini_models(st.session_state.gemini_key)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.gemini_models = []
            
            if st.session_state.gemini_models:
                col1, col2 = st.columns([4, 1])
                with col1:
                    current_model_index = 0
                    if st.session_state.gemini_model in st.session_state.gemini_models:
                        current_model_index = st.session_state.gemini_models.index(st.session_state.gemini_model)
                    st.session_state.gemini_model = st.selectbox(
                        "Model",
                        options=st.session_state.gemini_models,
                        index=current_model_index,
                        key="gemini_model_select"
                    )
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("🔄", key="refresh_gemini", help="Refresh"):
                        st.session_state.gemini_models = []
                        st.rerun()
            else:
                st.session_state.gemini_model = st.text_input(
                    "Model",
                    value=st.session_state.gemini_model,
                    key="gemini_model_input"
                )
            
            if st.session_state.gemini_key:
                provider_config = {
                    'provider': 'gemini',
                    'gemini_key': st.session_state.gemini_key,
                    'gemini_model': st.session_state.gemini_model
                }
        
        elif st.session_state.llm_provider == "ollama":
            new_host = st.text_input(
                "Ollama Host (optional)",
                value=st.session_state.ollama_host,
                placeholder="http://localhost:11434",
                key="ollama_host_input"
            )
            if new_host != st.session_state.ollama_host:
                st.session_state.ollama_models = []
            st.session_state.ollama_host = new_host

            if not st.session_state.ollama_models:
                with st.spinner("Fetching models..."):
                    try:
                        models = get_ollama_models(st.session_state.ollama_host or None)
                        st.session_state.ollama_models = models
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.ollama_models = []
            
            if st.session_state.ollama_models:
                col1, col2 = st.columns([4, 1])
                with col1:
                    current_model_index = 0
                    if st.session_state.ollama_model in st.session_state.ollama_models:
                        current_model_index = st.session_state.ollama_models.index(st.session_state.ollama_model)
                    st.session_state.ollama_model = st.selectbox(
                        "Model",
                        options=st.session_state.ollama_models,
                        index=current_model_index,
                        key="ollama_model_select"
                    )
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("🔄", key="refresh_ollama", help="Refresh"):
                        st.session_state.ollama_models = []
                        st.rerun()
            else:
                st.session_state.ollama_model = st.text_input(
                    "Model",
                    value=st.session_state.ollama_model,
                    key="ollama_model_input"
                )
            
            provider_config = {
                'provider': 'ollama',
                'ollama_model': st.session_state.ollama_model,
                'ollama_host': st.session_state.ollama_host
            }
        
        if st.session_state.llm_provider and provider_config:
            st.success(f"✅ {st.session_state.llm_provider.upper()} configured")

# Main content area
if st.session_state.comparison_data:
    # Comparison view
    st.header("📊 Stock Comparison")
    
    comparison = st.session_state.comparison_data
    
    if comparison['errors']:
        for error in comparison['errors']:
            st.warning(error)
    
    if comparison['comparison_table'] is not None and not comparison['comparison_table'].empty:
        # Display comparison table
        st.subheader("Comparison Table")
        df = comparison['comparison_table']
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Price comparison chart
            if 'Price (₹)' in df.columns:
                fig = px.bar(
                    df, 
                    x='Symbol', 
                    y='Price (₹)',
                    title="Price Comparison",
                    color='Price (₹)',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # P/E Ratio comparison
            if 'P/E' in df.columns:
                pe_data = df[df['P/E'] != 'N/A'].copy()
                if not pe_data.empty:
                    pe_data['P/E'] = pd.to_numeric(pe_data['P/E'], errors='coerce')
                    pe_data = pe_data.dropna(subset=['P/E'])
                    if not pe_data.empty:
                        fig = px.bar(
                            pe_data,
                            x='Symbol',
                            y='P/E',
                            title="P/E Ratio Comparison",
                            color='P/E',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison
        if '1Y Change (%)' in df.columns:
            perf_data = df[df['1Y Change (%)'] != 'N/A'].copy()
            if not perf_data.empty:
                perf_data['1Y Change (%)'] = pd.to_numeric(perf_data['1Y Change (%)'], errors='coerce')
                perf_data = perf_data.dropna(subset=['1Y Change (%)'])
                if not perf_data.empty:
                    fig = go.Figure()
                    colors = ['green' if x > 0 else 'red' for x in perf_data['1Y Change (%)']]
                    fig.add_trace(go.Bar(
                        x=perf_data['Symbol'],
                        y=perf_data['1Y Change (%)'],
                        marker_color=colors,
                        text=[f"{x:.2f}%" for x in perf_data['1Y Change (%)']],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="1 Year Performance Comparison",
                        yaxis_title="Change (%)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Chat for comparison
        st.divider()
        st.subheader("💬 Ask about the comparison")
        
        if prompt := st.chat_input("Ask a question about these stocks..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # Build combined data for analysis
                combined_data = {
                    'basic_info': {'name': f"Comparison: {', '.join(comparison['symbols'])}"},
                    'comparison_table': comparison['comparison_table'].to_dict('records'),
                    'stocks': comparison['stocks']
                }
                
                provider_config = None
                if st.session_state.llm_provider == "openai" and st.session_state.openai_key:
                    provider_config = {
                        'provider': 'openai',
                        'openai_key': st.session_state.openai_key,
                        'openai_model': st.session_state.openai_model
                    }
                elif st.session_state.llm_provider == "gemini" and st.session_state.gemini_key:
                    provider_config = {
                        'provider': 'gemini',
                        'gemini_key': st.session_state.gemini_key,
                        'gemini_model': st.session_state.gemini_model
                    }
                elif st.session_state.llm_provider == "ollama":
                    provider_config = {
                        'provider': 'ollama',
                        'ollama_model': st.session_state.ollama_model,
                        'ollama_host': st.session_state.ollama_host
                    }
                
                if st.session_state.enable_streaming and provider_config:
                    response_placeholder = st.empty()
                    full_response = ""
                    try:
                        for chunk in analyze_stock(combined_data, prompt, provider_config, stream=True):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        full_response = f"Error: {str(e)}"
                else:
                    response = analyze_stock(combined_data, prompt, provider_config, stream=False)
                    st.markdown(response)
                    full_response = response
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

elif st.session_state.current_symbol:
    # Single stock view with improved UI
    stock_data = st.session_state.stock_data
    basic_info = stock_data.get('basic_info', {})
    financials = stock_data.get('financials', {})
    ratios = stock_data.get('ratios', {})
    price_history = stock_data.get('price_history', {})
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{basic_info.get('current_price', 0):.2f}",
            delta=f"{price_history.get('1d_change', 0):.2f}%" if price_history.get('1d_change') else None
        )
    
    with col2:
        pe_ratio = ratios.get('pe', 0)
        st.metric(
            "P/E Ratio",
            f"{pe_ratio:.2f}" if pe_ratio else "N/A"
        )
    
    with col3:
        roe = financials.get('roe', 0)
        st.metric(
            "ROE",
            f"{roe:.2%}" if roe else "N/A"
        )
    
    with col4:
        market_cap = basic_info.get('market_cap', 0)
        st.metric(
            "Market Cap",
            f"₹{market_cap/1e7:.0f} Cr" if market_cap else "N/A"
        )
    
    # Performance chart
    if price_history:
        st.subheader("📈 Performance Metrics")
        perf_data = {
            'Period': ['1D', '1W', '1M', '3M', '6M', '1Y'],
            'Change (%)': [
                price_history.get('1d_change', 0),
                price_history.get('1w_change', 0),
                price_history.get('1m_change', 0),
                price_history.get('3m_change', 0),
                price_history.get('6m_change', 0),
                price_history.get('1y_change', 0)
            ]
        }
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df[perf_df['Change (%)'] != 0]
        
        if not perf_df.empty:
            fig = go.Figure()
            colors = ['green' if x > 0 else 'red' for x in perf_df['Change (%)']]
            fig.add_trace(go.Bar(
                x=perf_df['Period'],
                y=perf_df['Change (%)'],
                marker_color=colors,
                text=[f"{x:.2f}%" for x in perf_df['Change (%)']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Price Performance Over Different Periods",
                yaxis_title="Change (%)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Chat interface
    st.subheader("💬 Chat Analysis")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the stock..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with streaming
        with st.chat_message("assistant"):
            provider_config = None
            if st.session_state.llm_provider == "openai" and st.session_state.openai_key:
                provider_config = {
                    'provider': 'openai',
                    'openai_key': st.session_state.openai_key,
                    'openai_model': st.session_state.openai_model
                }
            elif st.session_state.llm_provider == "gemini" and st.session_state.gemini_key:
                provider_config = {
                    'provider': 'gemini',
                    'gemini_key': st.session_state.gemini_key,
                    'gemini_model': st.session_state.gemini_model
                }
            elif st.session_state.llm_provider == "ollama":
                provider_config = {
                    'provider': 'ollama',
                    'ollama_model': st.session_state.ollama_model,
                    'ollama_host': st.session_state.ollama_host
                }
            
            if st.session_state.enable_streaming and provider_config:
                # Streaming response
                response_placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in analyze_stock(stock_data, prompt, provider_config, stream=True):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    full_response = f"Error: {str(e)}"
            else:
                # Non-streaming response
                with st.spinner("Analyzing..."):
                    response = analyze_stock(stock_data, prompt, provider_config, stream=False)
                    st.markdown(response)
                    full_response = response
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    # Welcome screen
    st.info("👈 **Get Started:** Select a mode in the sidebar and enter stock symbols")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Single Stock Analysis
        - Enter a stock symbol
        - Get comprehensive analysis
        - Chat with AI about the stock
        - View key metrics and charts
        """)
    
    with col2:
        st.markdown("""
        ### 🔀 Stock Comparison
        - Compare 2-3 stocks side-by-side
        - View comparison tables
        - Analyze relative performance
        - Get AI insights on differences
        """)
