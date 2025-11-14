import json
import os
from datetime import datetime, timedelta
import hashlib


class StockDataCache:
    def __init__(self, cache_dir=".stock_cache", cache_duration_hours=1):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration_hours: How long to keep cached data (default 1 hour)
        """
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=cache_duration_hours)
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, symbol):
        """Generate cache key from stock symbol"""
        # Normalize symbol and create hash
        normalized = symbol.upper().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_cache_path(self, symbol):
        """Get cache file path for a symbol"""
        cache_key = self._get_cache_key(symbol)
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, symbol):
        """
        Get cached data for a symbol
        
        Returns:
            dict: Cached data if available and not expired, None otherwise
        """
        cache_path = self._get_cache_path(symbol)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cache_data.get('cached_at', ''))
            if datetime.now() - cached_time > self.cache_duration:
                # Cache expired, delete file
                os.remove(cache_path)
                return None
            
            # Return cached stock data
            return cache_data.get('data', None)
        
        except Exception as e:
            print(f"Error reading cache: {e}")
            # If cache file is corrupted, delete it
            try:
                os.remove(cache_path)
            except:
                pass
            return None
    
    def set(self, symbol, data):
        """
        Save data to cache
        
        Args:
            symbol: Stock symbol
            data: Stock data dictionary to cache
        """
        cache_path = self._get_cache_path(symbol)
        
        try:
            cache_data = {
                'symbol': symbol.upper().strip(),
                'cached_at': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error writing cache: {e}")
    
    def clear(self, symbol=None):
        """
        Clear cache for a specific symbol or all cache
        
        Args:
            symbol: Stock symbol to clear (None = clear all)
        """
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    return True
                except Exception as e:
                    print(f"Error clearing cache: {e}")
                    return False
        else:
            # Clear all cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, filename))
                return True
            except Exception as e:
                print(f"Error clearing all cache: {e}")
                return False
    
    def get_cache_info(self, symbol):
        """
        Get cache metadata (age, size, etc.)
        
        Returns:
            dict: Cache info or None if not cached
        """
        cache_path = self._get_cache_path(symbol)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cache_data.get('cached_at', ''))
            age = datetime.now() - cached_time
            file_size = os.path.getsize(cache_path)
            
            return {
                'cached_at': cached_time.isoformat(),
                'age_seconds': age.total_seconds(),
                'age_hours': age.total_seconds() / 3600,
                'file_size_kb': file_size / 1024,
                'is_expired': age > self.cache_duration
            }
        except Exception as e:
            print(f"Error getting cache info: {e}")
            return None
    
    def cleanup_expired(self):
        """Remove all expired cache files"""
        try:
            removed_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    cache_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        
                        cached_time = datetime.fromisoformat(cache_data.get('cached_at', ''))
                        if datetime.now() - cached_time > self.cache_duration:
                            os.remove(cache_path)
                            removed_count += 1
                    except:
                        # Corrupted file, remove it
                        try:
                            os.remove(cache_path)
                            removed_count += 1
                        except:
                            pass
            
            return removed_count
        except Exception as e:
            print(f"Error cleaning up cache: {e}")
            return 0

