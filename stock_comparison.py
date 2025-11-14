import pandas as pd
import streamlit as st
from data_fetcher import StockDataFetcher


def compare_stocks(symbols, force_refresh=False):
    """
    Compare multiple stocks and return comparison data
    
    Args:
        symbols: List of stock symbols to compare
        force_refresh: Whether to force refresh data
    
    Returns:
        dict: Comparison data with metrics for each stock
    """
    comparison_data = {
        'symbols': symbols,
        'stocks': {},
        'comparison_table': None,
        'errors': []
    }
    
    # Fetch data for each stock
    for symbol in symbols:
        try:
            fetcher = StockDataFetcher(symbol, use_cache=True, cache_duration_hours=1)
            data = fetcher.get_all_data(force_refresh=force_refresh)
            
            if data.get('basic_info'):
                comparison_data['stocks'][symbol.upper()] = data
            else:
                comparison_data['errors'].append(f"Could not fetch data for {symbol}")
        except Exception as e:
            comparison_data['errors'].append(f"Error fetching {symbol}: {str(e)}")
    
    # Create comparison table
    if comparison_data['stocks']:
        comparison_data['comparison_table'] = _create_comparison_table(comparison_data['stocks'])
    
    return comparison_data


def _create_comparison_table(stocks_data):
    """Create a pandas DataFrame for comparison"""
    rows = []
    
    for symbol, data in stocks_data.items():
        basic = data.get('basic_info', {})
        financials = data.get('financials', {})
        ratios = data.get('ratios', {})
        price_history = data.get('price_history', {})
        
        row = {
            'Symbol': symbol,
            'Name': basic.get('name', 'N/A'),
            'Sector': basic.get('sector', 'N/A'),
            'Price (₹)': round(basic.get('current_price', 0), 2),
            'Market Cap (₹ Cr)': round(basic.get('market_cap', 0) / 1e7, 2) if basic.get('market_cap') else 0,
            'P/E': round(ratios.get('pe', 0), 2) if ratios.get('pe') else 'N/A',
            'P/B': round(ratios.get('pb', 0), 2) if ratios.get('pb') else 'N/A',
            'ROE (%)': round(financials.get('roe', 0) * 100, 2) if financials.get('roe') else 'N/A',
            'Debt/Equity': round(financials.get('debt_to_equity', 0), 2) if financials.get('debt_to_equity') else 'N/A',
            'Div Yield (%)': round(ratios.get('dividend_yield', 0) * 100, 2) if ratios.get('dividend_yield') else 'N/A',
            '1D Change (%)': round(price_history.get('1d_change', 0), 2) if price_history.get('1d_change') else 'N/A',
            '1M Change (%)': round(price_history.get('1m_change', 0), 2) if price_history.get('1m_change') else 'N/A',
            '1Y Change (%)': round(price_history.get('1y_change', 0), 2) if price_history.get('1y_change') else 'N/A',
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_comparison_summary(comparison_data):
    """Generate a text summary of the comparison"""
    if not comparison_data['stocks']:
        return "No stock data available for comparison."
    
    summary = "## Stock Comparison Summary\n\n"
    
    for symbol, data in comparison_data['stocks'].items():
        basic = data.get('basic_info', {})
        ratios = data.get('ratios', {})
        price_history = data.get('price_history', {})
        
        summary += f"### {symbol} - {basic.get('name', 'N/A')}\n"
        summary += f"- **Price:** ₹{basic.get('current_price', 0):.2f}\n"
        summary += f"- **P/E Ratio:** {ratios.get('pe', 'N/A')}\n"
        summary += f"- **1Y Performance:** {price_history.get('1y_change', 0):.2f}%\n\n"
    
    return summary

