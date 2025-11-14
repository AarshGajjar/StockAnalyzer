import yfinance as yf
from datetime import datetime, timedelta
import json
import requests
from bs4 import BeautifulSoup
import time
import re
from cache_manager import StockDataCache

# Try importing optional dependencies
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class StockDataFetcher:
    def __init__(self, symbol, use_cache=True, cache_duration_hours=1):
        self.original_symbol = symbol.upper().strip()
        self.symbol = self._format_symbol(symbol)
        self.use_cache = use_cache
        self.cache = StockDataCache(cache_duration_hours=cache_duration_hours) if use_cache else None
        self.data = {
            'basic_info': {},
            'financials': {},
            'ratios': {},
            'news': [],
            'price_history': {},
            'peer_comparison': {},
            'data_sources': []
        }
    
    def _format_symbol(self, symbol):
        """Format symbol for yfinance (add .NS for NSE stocks)"""
        symbol = symbol.upper().strip()
        if '.' not in symbol:
            return f"{symbol}.NS"
        return symbol
    
    def _get_nse_symbol(self):
        """Get NSE symbol without suffix"""
        return self.original_symbol
    
    def fetch_basic_info(self):
        """Fetch basic stock information from multiple sources"""
        sources_used = []
        
        # Try yfinance first
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            if info and len(info) > 5:  # Valid data check
                self.data['basic_info'] = {
                    'name': info.get('longName', info.get('shortName', self.original_symbol)),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'currency': info.get('currency', 'INR'),
                    'volume': info.get('volume', 0),
                    '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                    '52_week_low': info.get('fiftyTwoWeekLow', 0),
                    'exchange': info.get('exchange', 'NSE')
                }
                sources_used.append('yfinance')
        except Exception as e:
            print(f"yfinance basic info error: {e}")
        
        # Try NSE India API
        try:
            nse_data = self._fetch_nse_data()
            if nse_data:
                # Merge NSE data
                if not self.data['basic_info'].get('current_price'):
                    self.data['basic_info']['current_price'] = nse_data.get('price', 0)
                if not self.data['basic_info'].get('name'):
                    self.data['basic_info']['name'] = nse_data.get('companyName', self.original_symbol)
                sources_used.append('nse')
        except Exception as e:
            print(f"NSE data error: {e}")
        
        self.data['data_sources'].extend(sources_used)
        return len(sources_used) > 0
    
    def _fetch_nse_data(self):
        """Fetch data from NSE India unofficial API"""
        try:
            symbol = self._get_nse_symbol()
            # NSE quote API
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': f'https://www.nseindia.com/get-quotes/equity?symbol={symbol}'
            }
            
            session = requests.Session()
            session.get('https://www.nseindia.com', headers=headers)  # Get cookies
            response = session.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                price_info = data.get('priceInfo', {})
                
                return {
                    'companyName': info.get('companyName', ''),
                    'price': price_info.get('lastPrice', 0),
                    'change': price_info.get('change', 0),
                    'changePercent': price_info.get('pChange', 0)
                }
        except Exception as e:
            print(f"NSE API error: {e}")
        return None
    
    def fetch_financials(self):
        """Fetch financial data from multiple sources"""
        sources_used = []
        
        # Try yfinance
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            hist = ticker.history(period="5y")
            
            financials = {
                'revenue': info.get('totalRevenue', 0),
                'profit': info.get('netIncomeToCommon', 0),
                'eps': info.get('trailingEps', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'ebitda': info.get('ebitda', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0)
            }
            
            # Calculate growth rates
            if len(hist) > 0:
                price_growth = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                financials['price_growth_5y'] = round(price_growth, 2)
            
            # Get financial statements if available
            try:
                financials_data = ticker.financials
                if not financials_data.empty:
                    # Get latest year data
                    latest_col = financials_data.columns[0]
                    financials['revenue_latest'] = financials_data.loc['Total Revenue', latest_col] if 'Total Revenue' in financials_data.index else financials.get('revenue', 0)
            except:
                pass
            
            self.data['financials'].update(financials)
            sources_used.append('yfinance')
        except Exception as e:
            print(f"yfinance financials error: {e}")
        
        # Try Screener.in scraping (lightweight)
        try:
            screener_data = self._fetch_screener_data()
            if screener_data:
                # Merge key metrics
                if not self.data['financials'].get('roe'):
                    self.data['financials']['roe'] = screener_data.get('roe', 0)
                if not self.data['financials'].get('debt_to_equity'):
                    self.data['financials']['debt_to_equity'] = screener_data.get('debt_to_equity', 0)
                sources_used.append('screener')
        except Exception as e:
            print(f"Screener error: {e}")
        
        self.data['data_sources'].extend(sources_used)
        return len(self.data['financials']) > 0
    
    def _fetch_screener_data(self):
        """Fetch basic data from Screener.in (lightweight scraping)"""
        try:
            symbol = self._get_nse_symbol()
            url = f"https://www.screener.in/company/{symbol}/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract key ratios from the page
                data = {}
                
                # Find ROE
                roe_elem = soup.find(string=re.compile('ROE'))
                if roe_elem:
                    parent = roe_elem.find_parent()
                    if parent:
                        value_text = parent.get_text()
                        roe_match = re.search(r'(\d+\.?\d*)%', value_text)
                        if roe_match:
                            data['roe'] = float(roe_match.group(1)) / 100
                
                # Find Debt to Equity
                debt_elem = soup.find(string=re.compile('Debt to Equity'))
                if debt_elem:
                    parent = debt_elem.find_parent()
                    if parent:
                        value_text = parent.get_text()
                        debt_match = re.search(r'(\d+\.?\d*)', value_text)
                        if debt_match:
                            data['debt_to_equity'] = float(debt_match.group(1))
                
                return data
        except Exception as e:
            print(f"Screener scraping error: {e}")
        return None
    
    def fetch_ratios(self):
        """Fetch valuation ratios from multiple sources"""
        sources_used = []
        
        # Try yfinance
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            ratios = {
                'pe': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb': info.get('priceToBook', 0),
                'ps': info.get('priceToSalesTrailing12Months', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0)
            }
            
            self.data['ratios'].update(ratios)
            sources_used.append('yfinance')
        except Exception as e:
            print(f"yfinance ratios error: {e}")
        
        self.data['data_sources'].extend(sources_used)
        return len(self.data['ratios']) > 0
    
    def fetch_news(self):
        """Fetch news from multiple sources"""
        all_news = []
        sources_used = []
        
        # Try yfinance news
        try:
            ticker = yf.Ticker(self.symbol)
            news = ticker.news[:10]  # Get more news items
            
            for item in news:
                # Extract date
                date_str = 'N/A'
                publish_time = item.get('providerPublishTime') or item.get('pubDate') or item.get('timestamp')
                if publish_time:
                    try:
                        if isinstance(publish_time, (int, float)) and publish_time > 0:
                            date_str = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d')
                        elif isinstance(publish_time, str):
                            # Try to parse string date
                            try:
                                date_obj = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                                date_str = date_obj.strftime('%Y-%m-%d')
                            except:
                                date_str = publish_time[:10] if len(publish_time) >= 10 else publish_time
                    except Exception as e:
                        print(f"Date parsing error: {e}")
                
                # Extract summary
                summary = item.get('summary', '') or item.get('description', '') or item.get('text', '')
                if summary:
                    summary = summary[:300].strip()
                
                # Extract title
                title = item.get('title', 'N/A') or item.get('headline', 'N/A')
                
                # Extract URL
                url = item.get('link', '') or item.get('url', '')
                
                # Extract source
                source = item.get('publisher', 'Yahoo Finance') or item.get('source', 'Yahoo Finance')
                
                all_news.append({
                    'title': title,
                    'date': date_str,
                    'summary': summary,
                    'source': source,
                    'url': url
                })
            sources_used.append('yfinance')
        except Exception as e:
            print(f"yfinance news error: {e}")
        
        # Try Moneycontrol news
        try:
            moneycontrol_news = self._fetch_moneycontrol_news()
            if moneycontrol_news:
                all_news.extend(moneycontrol_news)
                sources_used.append('moneycontrol')
        except Exception as e:
            print(f"Moneycontrol news error: {e}")
        
        # Try NSE announcements
        try:
            nse_news = self._fetch_nse_announcements()
            if nse_news:
                all_news.extend(nse_news)
                sources_used.append('nse_announcements')
        except Exception as e:
            print(f"NSE announcements error: {e}")
        
        # Remove duplicates and sort by date
        seen_titles = set()
        unique_news = []
        for item in all_news:
            title_lower = item['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_news.append(item)
        
        # Sort by date (newest first)
        unique_news.sort(key=lambda x: x['date'], reverse=True)
        
        self.data['news'] = unique_news[:15]  # Top 15 news items
        self.data['data_sources'].extend(sources_used)
        return len(self.data['news']) > 0
    
    def _fetch_moneycontrol_news(self):
        """Fetch news from Moneycontrol"""
        try:
            symbol = self._get_nse_symbol()
            # Moneycontrol uses company name, try symbol first
            url = f"https://www.moneycontrol.com/news/tags/{symbol.lower()}.html"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                news_items = []
                
                # Try multiple selectors for articles
                articles = soup.find_all('li', class_='clearfix')[:10]
                if not articles:
                    articles = soup.find_all('div', class_='news_list')[:10]
                if not articles:
                    articles = soup.find_all('article')[:10]
                
                for article in articles:
                    # Find title
                    title_elem = article.find('h2') or article.find('h3') or article.find('a', class_='title')
                    link_elem = article.find('a')
                    
                    if not title_elem and link_elem:
                        title_elem = link_elem
                    
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        if not title:
                            continue
                        
                        article_url = link_elem.get('href', '')
                        if not article_url.startswith('http'):
                            article_url = f"https://www.moneycontrol.com{article_url}"
                        
                        # Find date - try multiple selectors
                        date_elem = (article.find('span', class_='date') or 
                                   article.find('span', class_='time') or
                                   article.find('time') or
                                   article.find('div', class_='date'))
                        date_str = 'N/A'
                        if date_elem:
                            date_text = date_elem.get_text(strip=True)
                            # Try to parse common date formats
                            if date_text:
                                # Clean up date string
                                date_str = date_text.replace('Updated:', '').replace('Published:', '').strip()
                                # Try to extract date from text like "2 hours ago", "Jan 15, 2024"
                                if not any(char.isdigit() for char in date_str):
                                    date_str = date_text  # Keep original if no digits
                        
                        # Try to get summary/description
                        summary = ''
                        desc_elem = (article.find('p') or 
                                    article.find('div', class_='desc') or
                                    article.find('div', class_='summary'))
                        if desc_elem:
                            summary = desc_elem.get_text(strip=True)[:300]
                        
                        news_items.append({
                            'title': title,
                            'date': date_str if date_str else 'N/A',
                            'summary': summary,
                            'source': 'Moneycontrol',
                            'url': article_url
                        })
                
                return news_items
        except Exception as e:
            print(f"Moneycontrol scraping error: {e}")
        return None
    
    def _fetch_nse_announcements(self):
        """Fetch corporate announcements from NSE"""
        try:
            symbol = self._get_nse_symbol()
            url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': f'https://www.nseindia.com/companies-listing/corporate-filings-announcements?symbol={symbol}'
            }
            
            session = requests.Session()
            session.get('https://www.nseindia.com', headers=headers)
            response = session.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                announcements = []
                
                for item in data.get('data', [])[:10]:
                    # Extract date - try multiple fields
                    date_str = (item.get('date') or 
                              item.get('announcementDate') or 
                              item.get('publishedDate') or 
                              item.get('timestamp') or 'N/A')
                    
                    # Format date if it's a timestamp
                    if isinstance(date_str, (int, float)) and date_str > 0:
                        try:
                            date_str = datetime.fromtimestamp(date_str / 1000 if date_str > 1e10 else date_str).strftime('%Y-%m-%d')
                        except:
                            date_str = str(date_str)
                    elif isinstance(date_str, str) and len(date_str) > 10:
                        # Try to extract date from string
                        try:
                            # Common formats: "2024-01-15T10:30:00" or "15 Jan 2024"
                            if 'T' in date_str:
                                date_str = date_str.split('T')[0]
                            elif len(date_str) >= 10:
                                date_str = date_str[:10]
                        except:
                            pass
                    
                    # Extract summary/description
                    summary = (item.get('desc') or 
                             item.get('description') or 
                             item.get('summary') or 
                             item.get('body', ''))
                    if summary:
                        # Clean HTML if present
                        if '<' in summary:
                            soup = BeautifulSoup(summary, 'html.parser')
                            summary = soup.get_text(strip=True)
                        summary = summary[:300]
                    
                    # Extract title
                    title = (item.get('subject') or 
                           item.get('title') or 
                           item.get('headline') or 'N/A')
                    
                    # Extract URL if available
                    url = (item.get('url') or 
                          item.get('link') or 
                          item.get('attachmentUrl') or '')
                    
                    announcements.append({
                        'title': title,
                        'date': date_str if date_str else 'N/A',
                        'summary': summary if summary else '',
                        'source': 'NSE Announcements',
                        'url': url
                    })
                
                return announcements
        except Exception as e:
            print(f"NSE announcements error: {e}")
        return None
    
    def fetch_price_history(self):
        """Fetch historical price data"""
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                self.data['price_history'] = {
                    '1d_change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0,
                    '1w_change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-6]) - 1) * 100 if len(hist) > 6 else 0,
                    '1m_change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 21 else 0,
                    '3m_change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-63]) - 1) * 100 if len(hist) > 63 else 0,
                    '6m_change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-126]) - 1) * 100 if len(hist) > 126 else 0,
                    '1y_change': ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100,
                    'volatility': hist['Close'].pct_change().std() * 100 if len(hist) > 1 else 0
                }
                return True
        except Exception as e:
            print(f"Price history error: {e}")
        return False
    
    def get_all_data(self, force_refresh=False):
        """
        Fetch all available data from multiple sources
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh data
        
        Returns:
            dict: Stock data dictionary
        """
        # Check cache first
        if self.use_cache and self.cache and not force_refresh:
            cached_data = self.cache.get(self.original_symbol)
            if cached_data:
                print(f"Using cached data for {self.original_symbol}")
                self.data = cached_data
                # Add cache indicator
                self.data['_from_cache'] = True
                cache_info = self.cache.get_cache_info(self.original_symbol)
                if cache_info:
                    self.data['_cache_age_hours'] = round(cache_info['age_hours'], 2)
                return self.data
        
        print(f"Fetching fresh data for {self.original_symbol} from multiple sources...")
        
        # Reset data
        self.data = {
            'basic_info': {},
            'financials': {},
            'ratios': {},
            'news': [],
            'price_history': {},
            'peer_comparison': {},
            'data_sources': []
        }
        
        # Fetch in parallel where possible
        self.fetch_basic_info()
        self.fetch_financials()
        self.fetch_ratios()
        self.fetch_news()
        self.fetch_price_history()
        
        # Remove duplicate sources
        self.data['data_sources'] = list(set(self.data['data_sources']))
        
        # Mark as fresh data
        self.data['_from_cache'] = False
        
        # Save to cache
        if self.use_cache and self.cache:
            self.cache.set(self.original_symbol, self.data)
            print(f"Cached data for {self.original_symbol}")
        
        return self.data
