import ccxt
import time
from settings import OKX_API_KEY, OKX_SECRET, OKX_PASSWORD, OKX_TESTNET


def create_exchange():
    """Create OKX exchange instance"""
    e = ccxt.okx({
        "apiKey": OKX_API_KEY,
        "secret": OKX_SECRET,
        "password": OKX_PASSWORD,
        "enableRateLimit": True,
        "rateLimit": 100,
        "timeout": 30000,
        "options": {
            'defaultType': 'spot',
            'recvWindow': 10000,
        }
    })
    e.set_sandbox_mode(OKX_TESTNET)
    return e


class ExchangeManager:
    """Manages exchange connections and API rate limiting"""
    
    def __init__(self):
        self.exchange = create_exchange()
        self.max_retries = 3
        self.retry_delay = 2.0
        self.last_api_call_time = 0
        self.min_api_interval = 0.5
        self.api_call_count = 0
        self.api_rate_limit_count = 0
        
        try:
            self.exchange.load_markets()
        except Exception as e:
            print(f"Market data loading failed, will retry on first use: {e}")
    
    def safe_api_call(self, func, *args, **kwargs):
        """API call with retry mechanism and rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_api_interval:
            sleep_time = self.min_api_interval - time_since_last_call
            time.sleep(sleep_time)
        
        for attempt in range(self.max_retries):
            try:
                self.last_api_call_time = time.time()
                self.api_call_count += 1
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "50011" in error_msg:
                    self.api_rate_limit_count += 1
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        print(f"API rate limit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries}) [total: {self.api_call_count}, limits: {self.api_rate_limit_count}]")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"API retry failed: {error_msg} [total: {self.api_call_count}, limits: {self.api_rate_limit_count}]")
                        return None
                else:
                    raise e

