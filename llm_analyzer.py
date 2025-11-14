import os
import json
from dotenv import load_dotenv

load_dotenv()

# Try importing LLM libraries (optional dependencies)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Global clients (will be set dynamically)
openai_client = None
gemini_client = None
gemini_model_name = None
ollama_model = "llama2"

SYSTEM_PROMPT = """You are a stock analysis assistant specializing in Indian equities. 
You provide fundamental analysis based on financial data.

Always cite specific metrics when making claims. Be objective and mention both positives and negatives.
Keep responses concise but informative. If data is missing, acknowledge it."""


def _format_comparison_context(comparison_data):
    """Format context for stock comparison"""
    import pandas as pd
    
    comparison_table = comparison_data.get('comparison_table')
    stocks = comparison_data.get('stocks', {})
    
    context = "Stock Comparison Data:\n\n"
    
    # Handle DataFrame
    if comparison_table is not None:
        if isinstance(comparison_table, pd.DataFrame):
            # Convert DataFrame to dict for JSON serialization
            context += "Comparison Table:\n"
            for _, row in comparison_table.iterrows():
                symbol = row.get('Symbol', 'N/A')
                context += f"\n{symbol}:\n"
                context += f"  Name: {row.get('Name', 'N/A')}\n"
                context += f"  Sector: {row.get('Sector', 'N/A')}\n"
                context += f"  Price: ₹{row.get('Price (₹)', 'N/A')}\n"
                context += f"  Market Cap: ₹{row.get('Market Cap (₹ Cr)', 'N/A')} Cr\n"
                context += f"  P/E: {row.get('P/E', 'N/A')}\n"
                context += f"  P/B: {row.get('P/B', 'N/A')}\n"
                context += f"  ROE: {row.get('ROE (%)', 'N/A')}%\n"
                context += f"  Debt/Equity: {row.get('Debt/Equity', 'N/A')}\n"
                context += f"  Dividend Yield: {row.get('Div Yield (%)', 'N/A')}%\n"
                context += f"  1D Change: {row.get('1D Change (%)', 'N/A')}%\n"
                context += f"  1M Change: {row.get('1M Change (%)', 'N/A')}%\n"
                context += f"  1Y Change: {row.get('1Y Change (%)', 'N/A')}%\n"
        elif isinstance(comparison_table, list):
            # Handle list format
            context += "Comparison Table:\n"
            for row in comparison_table:
                context += f"- {row.get('Symbol', 'N/A')}: "
                context += f"Price: ₹{row.get('Price (₹)', 'N/A')}, "
                context += f"P/E: {row.get('P/E', 'N/A')}\n"
    
    # Add detailed info from stocks dict
    if stocks:
        context += "\nDetailed Stock Information:\n"
        for symbol, data in stocks.items():
            basic = data.get('basic_info', {})
            ratios = data.get('ratios', {})
            financials = data.get('financials', {})
            price_history = data.get('price_history', {})
            
            context += f"\n{symbol} - {basic.get('name', 'N/A')}:\n"
            context += f"  Price: ₹{basic.get('current_price', 0):.2f}\n"
            context += f"  Market Cap: ₹{basic.get('market_cap', 0):,.0f}\n"
            context += f"  P/E: {ratios.get('pe', 'N/A')}\n"
            context += f"  ROE: {financials.get('roe', 0):.2% if financials.get('roe') else 'N/A'}\n"
            context += f"  1Y Performance: {price_history.get('1y_change', 0):.2f}%\n"
    
    return context


def get_openai_models(api_key):
    """Fetch available OpenAI models"""
    if not OPENAI_AVAILABLE:
        return []
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        # Filter for chat completion models
        chat_models = [m.id for m in models.data if 'gpt' in m.id.lower() and ('turbo' in m.id.lower() or 'gpt-4' in m.id.lower())]
        # Sort and remove duplicates
        chat_models = sorted(list(set(chat_models)))
        return chat_models
    except Exception as e:
        raise Exception(f"Failed to fetch OpenAI models: {str(e)}")


def get_gemini_models(api_key):
    """Fetch available Gemini models"""
    if not GEMINI_AVAILABLE:
        return []
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter for models that support generateContent
        gemini_models = [m.name.split('/')[-1] for m in models 
                        if 'generateContent' in m.supported_generation_methods 
                        and 'gemini' in m.name.lower()]
        return sorted(gemini_models)
    except Exception as e:
        raise Exception(f"Failed to fetch Gemini models: {str(e)}")


def get_ollama_models():
    """Fetch available Ollama models"""
    if not OLLAMA_AVAILABLE:
        return []
    try:
        response = ollama.list()
        
        # Handle different response structures
        if isinstance(response, dict):
            models_list = response.get('models', [])
        elif isinstance(response, list):
            models_list = response
        else:
            # Try to access as attribute
            models_list = getattr(response, 'models', [])
        
        model_names = []
        for model in models_list:
            # Handle different model structures
            if isinstance(model, dict):
                # Try different possible keys
                name = model.get('name') or model.get('model') or model.get('model_name')
            elif isinstance(model, str):
                name = model
            else:
                # Try as object attribute
                name = getattr(model, 'name', None) or getattr(model, 'model', None)
            
            if name:
                # Clean up model name (remove tags if present)
                if ':' in name:
                    name = name.split(':')[0]
                model_names.append(name)
        
        return sorted(list(set(model_names)))  # Remove duplicates and sort
    except Exception as e:
        raise Exception(f"Failed to fetch Ollama models: {str(e)}")


def test_provider_connection(provider_config):
    """Test connection to a provider with the given configuration"""
    try:
        initialize_providers(provider_config)
        provider = provider_config.get('provider')
        
        if provider == 'openai':
            # Test with a simple completion
            test_response = openai_client.chat.completions.create(
                model=provider_config.get('openai_model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=5
            )
            return True, "Connection successful!"
        
        elif provider == 'gemini':
            # Test with a simple generation
            test_model = genai.GenerativeModel(provider_config.get('gemini_model', 'gemini-1.5-flash'))
            test_response = test_model.generate_content("Say test")
            return True, "Connection successful!"
        
        elif provider == 'ollama':
            # Test with a simple chat
            test_response = ollama.chat(
                model=provider_config.get('ollama_model', 'llama2'),
                messages=[{"role": "user", "content": "Say test"}]
            )
            return True, "Connection successful!"
        
        return False, "Unknown provider"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def initialize_providers(provider_config):
    """Initialize LLM providers based on UI configuration"""
    global openai_client, gemini_client, gemini_model_name, ollama_model
    
    # Reset clients
    openai_client = None
    gemini_client = None
    gemini_model_name = None
    
    # Initialize OpenAI
    if provider_config.get('provider') == 'openai':
        openai_key = provider_config.get('openai_key', '').strip()
        if OPENAI_AVAILABLE and openai_key:
            try:
                openai_client = OpenAI(api_key=openai_key)
            except Exception as e:
                raise Exception(f"Failed to initialize OpenAI: {e}")
        elif not OPENAI_AVAILABLE:
            raise Exception("OpenAI library not installed. Run: pip install openai")
        elif not openai_key:
            raise Exception("OpenAI API key is required")
    
    # Initialize Gemini
    elif provider_config.get('provider') == 'gemini':
        gemini_key = provider_config.get('gemini_key', '').strip()
        gemini_model_name = provider_config.get('gemini_model', 'gemini-1.5-flash')
        if GEMINI_AVAILABLE and gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                gemini_client = genai.GenerativeModel(gemini_model_name)
            except Exception as e:
                raise Exception(f"Failed to initialize Gemini: {e}")
        elif not GEMINI_AVAILABLE:
            raise Exception("Gemini library not installed. Run: pip install google-generativeai")
        elif not gemini_key:
            raise Exception("Gemini API key is required")
    
    # Initialize Ollama
    elif provider_config.get('provider') == 'ollama':
        ollama_model = provider_config.get('ollama_model', 'llama2')
        if not OLLAMA_AVAILABLE:
            raise Exception("Ollama library not installed. Run: pip install ollama")
        # Test if Ollama is running
        try:
            ollama.list()  # Test connection
        except Exception as e:
            raise Exception(f"Ollama is not running or not accessible: {e}")


def _get_provider(provider_config=None):
    """Determine which LLM provider to use"""
    if provider_config:
        return provider_config.get('provider')
    
    # Fallback to env-based detection (for backward compatibility)
    if openai_client:
        return "openai"
    elif gemini_client:
        return "gemini"
    elif OLLAMA_AVAILABLE:
        return "ollama"
    return None


def get_provider_status(provider_config=None):
    """Get status of all LLM providers (for debugging)"""
    status = {
        "openai": {
            "available": OPENAI_AVAILABLE,
            "configured": openai_client is not None,
            "has_key": bool(provider_config and provider_config.get('openai_key', '').strip()) if provider_config else False
        },
        "gemini": {
            "available": GEMINI_AVAILABLE,
            "configured": gemini_client is not None,
            "has_key": bool(provider_config and provider_config.get('gemini_key', '').strip()) if provider_config else False,
            "model": provider_config.get('gemini_model') if provider_config else gemini_model_name
        },
        "ollama": {
            "available": OLLAMA_AVAILABLE,
            "configured": provider_config and provider_config.get('provider') == 'ollama' if provider_config else False,
            "model": provider_config.get('ollama_model') if provider_config and provider_config.get('provider') == 'ollama' else None
        }
    }
    return status


def analyze_stock(stock_data, user_question, provider_config=None, stream=False):
    """Analyze stock data and answer user question using LLM
    
    Args:
        stock_data: Stock data dictionary
        user_question: User's question
        provider_config: LLM provider configuration
        stream: If True, returns a generator for streaming responses
    
    Returns:
        If stream=True: Generator yielding response chunks
        If stream=False: Complete response string
    """
    
    # Initialize provider if config provided
    if provider_config and provider_config.get('provider'):
        try:
            initialize_providers(provider_config)
        except Exception as e:
            error_msg = f"❌ Error initializing LLM provider: {str(e)}\n\n{_simple_analysis(stock_data, user_question)}"
            if stream:
                yield error_msg
                return
            return error_msg
    
    provider = _get_provider(provider_config)
    
    if not provider:
        # Fallback to simple analysis if no LLM available
        result = _simple_analysis(stock_data, user_question)
        result += "\n\n💡 **Tip:** Configure an LLM provider in the sidebar to get AI-powered analysis."
        if stream:
            yield result
            return
        return result
    
    try:
        # Check if this is comparison data
        if 'comparison_table' in stock_data or 'stocks' in stock_data:
            # Handle comparison data
            context = _format_comparison_context(stock_data)
            user_prompt = f"{context}\n\nQuestion: {user_question}"
        else:
            # Handle single stock data
            basic_info = stock_data.get('basic_info', {})
            financials = stock_data.get('financials', {})
            ratios = stock_data.get('ratios', {})
            price_history = stock_data.get('price_history', {})
            data_sources = stock_data.get('data_sources', [])
        
            # Format percentage values
            roe_val = financials.get('roe', 0)
            roe_str = f"{roe_val:.2%}" if roe_val else 'N/A'
            
            roa_val = financials.get('roa', 0)
            roa_str = f"{roa_val:.2%}" if roa_val else 'N/A'
            
            profit_margin_val = financials.get('profit_margin', 0)
            profit_margin_str = f"{profit_margin_val:.2%}" if profit_margin_val else 'N/A'
            
            operating_margin_val = financials.get('operating_margin', 0)
            operating_margin_str = f"{operating_margin_val:.2%}" if operating_margin_val else 'N/A'
            
            div_yield = ratios.get('dividend_yield', 0)
            div_yield_str = f"{div_yield:.2%}" if div_yield else 'N/A'
            
            # Format optional numeric values
            forward_pe = ratios.get('forward_pe', 0)
            forward_pe_str = f"{forward_pe:.2f}" if forward_pe else 'N/A'
            
            ps_ratio = ratios.get('ps', 0)
            ps_str = f"{ps_ratio:.2f}" if ps_ratio else 'N/A'
            
            peg_ratio = ratios.get('peg_ratio', 0)
            peg_str = f"{peg_ratio:.2f}" if peg_ratio else 'N/A'
            
            current_ratio = ratios.get('current_ratio', 0)
            current_ratio_str = f"{current_ratio:.2f}" if current_ratio else 'N/A'
            
            ev_revenue = ratios.get('ev_to_revenue', 0)
            ev_revenue_str = f"{ev_revenue:.2f}" if ev_revenue else 'N/A'
            
            ev_ebitda = ratios.get('ev_to_ebitda', 0)
            ev_ebitda_str = f"{ev_ebitda:.2f}" if ev_ebitda else 'N/A'
            
            # Format context from stock data
            context = f"""
Stock Data for {basic_info.get('name', 'Unknown')} ({basic_info.get('exchange', 'NSE')}):

Basic Info:
- Sector: {basic_info.get('sector', 'N/A')}
- Industry: {basic_info.get('industry', 'N/A')}
- Market Cap: {basic_info.get('market_cap', 0):,.0f} {basic_info.get('currency', 'INR')}
- Current Price: {basic_info.get('current_price', 0):.2f}
- 52 Week High: {basic_info.get('52_week_high', 0):.2f}
- 52 Week Low: {basic_info.get('52_week_low', 0):.2f}
- Volume: {basic_info.get('volume', 0):,.0f}

Financials:
- Revenue: {financials.get('revenue', 0):,.0f}
- Profit: {financials.get('profit', 0):,.0f}
- EPS: {financials.get('eps', 0):.2f}
- ROE: {roe_str}
- ROA: {roa_str}
- Debt to Equity: {financials.get('debt_to_equity', 0):.2f}
- Profit Margin: {profit_margin_str}
- Operating Margin: {operating_margin_str}
- EBITDA: {financials.get('ebitda', 0):,.0f}
- Total Debt: {financials.get('total_debt', 0):,.0f}
- Total Cash: {financials.get('total_cash', 0):,.0f}
- 5Y Price Growth: {financials.get('price_growth_5y', 0):.2f}%

Valuation Ratios:
- P/E Ratio: {ratios.get('pe', 0):.2f}
- Forward P/E: {forward_pe_str}
- P/B Ratio: {ratios.get('pb', 0):.2f}
- P/S Ratio: {ps_str}
- PEG Ratio: {peg_str}
- Dividend Yield: {div_yield_str}
- Current Ratio: {current_ratio_str}
- EV/Revenue: {ev_revenue_str}
- EV/EBITDA: {ev_ebitda_str}

Price Performance:
- 1 Day: {price_history.get('1d_change', 0):.2f}%
- 1 Week: {price_history.get('1w_change', 0):.2f}%
- 1 Month: {price_history.get('1m_change', 0):.2f}%
- 3 Months: {price_history.get('3m_change', 0):.2f}%
- 6 Months: {price_history.get('6m_change', 0):.2f}%
- 1 Year: {price_history.get('1y_change', 0):.2f}%
- Volatility: {price_history.get('volatility', 0):.2f}%

Recent News ({len(stock_data.get('news', []))} items):
{json.dumps(stock_data.get('news', [])[:5], indent=2)}

Data Sources: {', '.join(data_sources) if data_sources else 'N/A'}
"""
            user_prompt = f"{context}\n\nQuestion: {user_question}"
        
        if provider == "openai":
            model_name = provider_config.get('openai_model') if provider_config else None
            if stream:
                stream_gen = _analyze_with_openai(user_prompt, model_name, stream=True)
                for chunk in stream_gen:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            else:
                return _analyze_with_openai(user_prompt, model_name, stream=False)
        elif provider == "gemini":
            # Gemini doesn't support streaming in the same way, return full response
            if stream:
                response = _analyze_with_gemini(user_prompt)
                # Simulate streaming by yielding chunks
                words = response.split()
                for i in range(0, len(words), 3):
                    yield ' '.join(words[i:i+3]) + ' '
                return
            else:
                return _analyze_with_gemini(user_prompt)
        elif provider == "ollama":
            model_name = provider_config.get('ollama_model') if provider_config else None
            if stream:
                stream_gen = _analyze_with_ollama(user_prompt, model_name, stream=True)
                for chunk in stream_gen:
                    if chunk.get('message', {}).get('content'):
                        yield chunk['message']['content']
                return
            else:
                return _analyze_with_ollama(user_prompt, model_name, stream=False)
        
    except Exception as e:
        error_msg = f"Error analyzing stock: {str(e)}\n\n{_simple_analysis(stock_data, user_question)}"
        if stream:
            yield error_msg
            return
        return error_msg


def _analyze_with_openai(user_prompt, model_name=None, stream=False):
    """Analyze using OpenAI API"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    model = model_name or "gpt-3.5-turbo"
    
    if stream:
        # Return generator for streaming
        stream_response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )
        return stream_response
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content


def _analyze_with_gemini(user_prompt):
    """Analyze using Google Gemini API"""
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    
    try:
        response = gemini_client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        # If model fails, try to use a different one
        error_msg = str(e)
        if "not found" in error_msg.lower() or "not supported" in error_msg.lower():
            # Try to list available models and use the first one
            try:
                available_models = [m.name.split('/')[-1] for m in genai.list_models() 
                                  if 'generateContent' in m.supported_generation_methods]
                if available_models:
                    new_model = genai.GenerativeModel(available_models[0])
                    response = new_model.generate_content(full_prompt)
                    return response.text
            except:
                pass
        raise e


def _analyze_with_ollama(user_prompt, model_name=None, stream=False):
    """Analyze using Ollama (local LLM)"""
    model = model_name or ollama_model
    
    if stream:
        # Ollama supports streaming
        stream_response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )
        return stream_response
    else:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response['message']['content']


def _simple_analysis(stock_data, user_question):
    """Simple rule-based analysis fallback when LLM is not available"""
    basic = stock_data.get('basic_info', {})
    financials = stock_data.get('financials', {})
    ratios = stock_data.get('ratios', {})
    
    response = f"Analysis for {basic.get('name', 'Stock')}:\n\n"
    
    # Basic valuation check
    pe = ratios.get('pe', 0)
    if pe > 0:
        if pe < 15:
            response += f"• P/E ratio of {pe:.2f} suggests reasonable valuation.\n"
        elif pe > 25:
            response += f"• P/E ratio of {pe:.2f} indicates high valuation.\n"
        else:
            response += f"• P/E ratio of {pe:.2f} is moderate.\n"
    
    # ROE check
    roe = financials.get('roe', 0)
    if roe > 0:
        if roe > 0.15:
            response += f"• ROE of {roe:.1%} shows strong profitability.\n"
        else:
            response += f"• ROE of {roe:.1%} indicates moderate profitability.\n"
    
    # Debt check
    debt_equity = financials.get('debt_to_equity', 0)
    if debt_equity > 0:
        if debt_equity > 1:
            response += f"• Debt-to-equity of {debt_equity:.2f} suggests high leverage.\n"
        else:
            response += f"• Debt-to-equity of {debt_equity:.2f} indicates manageable debt.\n"
    
    provider = _get_provider()
    if not provider:
        response += f"\nNote: For detailed analysis, please set OPENAI_API_KEY, GEMINI_API_KEY, or USE_OLLAMA=true in .env file."
    
    return response

