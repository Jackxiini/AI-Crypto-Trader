from typing import Optional, Dict, List, Tuple
import time
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import openai
import google.generativeai as genai

from core.exchange import create_exchange
from models import (
    LLMTradingInput, LLMTradingOutput, 
    Prices, TrendFilters, Volatility, StructureSignals, 
    LiquidityCosts, RiskRules, Precalc, PositionState,
    Entry, StopLoss, TakeProfit, Checklist,
    EnhancedPrices, TechnicalIndicators, MarketStructure,
    OrderBookData, MarketSentiment, TimeBasedData, EnhancedLLMTradingInput
)


class LLMTradingSystem:
    def __init__(self):
        self.exchange = create_exchange()
        # Lazy load market data to avoid rate limiting on startup
        try:
            self.exchange.load_markets()
        except Exception as e:
            print(f"Market data loading failed, will retry on first use: {e}")
        
        # Supported trading pairs
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'SOL/USDT', 'DOGE/USDT']
        
        # API request retry configuration
        self.max_retries = 3
        self.retry_delay = 2.0  # Initial delay 2 seconds
        self.last_api_call_time = 0  # Last API call time
        self.min_api_interval = 0.5  # Minimum API call interval (seconds) - increased to 500ms
        self.api_call_count = 0  # API call counter
        self.api_rate_limit_count = 0  # Rate limit counter
        
        # Trading parameters
        self.min_order_amount = 5.0  # Minimum order amount (USDT)
        self.max_position_size = 200.0  # Maximum single position size (USDT)
        self.default_stop_loss_pct = 0.05  # Default stop loss percentage (5%)
        self.default_take_profit_pct = 0.1  # Default take profit percentage (10%)
        
        # Trading plan status
        self.trading_plan_active = False
        self.selected_llm = 'gemini'  # Default to gemini as GPT is more expensive
        self.current_positions = {}  # Current positions
        self.pending_orders = {}  # Pending orders
        self.analysis_history = []  # Analysis history (max 100 records)
        self.max_history_size = 100
        
        # LLM API configuration
        self.openai_api_key = None
        self.gemini_api_key = None
        self.deepseek_api_key = None
        
        # Technical indicator parameters
        self.rsi_period = 14
        self.ma_short = 5
        self.ma_long = 20
        self.bb_period = 20
        self.bb_std = 2
    
    def safe_api_call(self, func, *args, **kwargs):
        """API call with retry mechanism and rate limiting"""
        # Ensure API call interval
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
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"API rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries}) [, total calls: {self.api_call_count}, , rate limits: {self.api_rate_limit_count}]")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"API retry failed, max retries reached: {error_msg} [, total calls: {self.api_call_count}, , rate limits: {self.api_rate_limit_count}]")
                        # For rate limit, return None instead of raising exception
                        return None
                else:
                    # Non-rate limit errors raise directly
                    raise e
        
    def get_kline_data(self, symbol: str, timeframe: str = '3m', limit: int = 100) -> pd.DataFrame:
        """Get K-line data"""
        try:
            ohlcv = self.safe_api_call(self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Failed to get K-line data {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_enhanced_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate enhanced technical indicators"""
        try:
            # RSI indicator
            rsi_14 = self.calculate_rsi(df['close'], 14).iloc[-1]
            rsi_21 = self.calculate_rsi(df['close'], 21).iloc[-1]
            
            # MACD indicator
            macd_line, macd_signal, macd_histogram = self.calculate_macd(df['close'])
            
            # Moving averages
            sma_5 = df['close'].rolling(5).mean().iloc[-1]
            sma_10 = df['close'].rolling(10).mean().iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            ema_5 = df['close'].ewm(span=5).mean().iloc[-1]
            ema_10 = df['close'].ewm(span=10).mean().iloc[-1]
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_100 = df['close'].ewm(span=100).mean().iloc[-1] if len(df) >= 100 else None
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) >= 200 else None
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else None
            bb_position = (df['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else None
            
            # Volume indicators
            volume_sma_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / volume_sma_20 if volume_sma_20 > 0 else None
            obv = self.calculate_obv(df)
            
            # ATR indicator
            atr_14 = self.calculate_atr(df, 14)
            atr_21 = self.calculate_atr(df, 21)
            
            # Stochastic indicator
            stochastic_k, stochastic_d = self.calculate_stochastic(df)
            
            # Williams %R
            williams_r = self.calculate_williams_r(df)
            
            # CCI indicator
            cci = self.calculate_cci(df)
            
            # ADX indicator
            adx, di_plus, di_minus = self.calculate_adx(df)
            
            return TechnicalIndicators(
                rsi_14=rsi_14,
                rsi_21=rsi_21,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                sma_5=sma_5,
                sma_10=sma_10,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_5=ema_5,
                ema_10=ema_10,
                ema_20=ema_20,
                ema_50=ema_50,
                ema_100=ema_100,
                ema_200=ema_200,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                bb_width=bb_width,
                bb_position=bb_position,
                volume_sma_20=volume_sma_20,
                volume_ratio=volume_ratio,
                obv=obv,
                atr_14=atr_14,
                atr_21=atr_21,
                bollinger_width=bb_width,
                stochastic_k=stochastic_k,
                stochastic_d=stochastic_d,
                williams_r=williams_r,
                cci=cci,
                adx=adx,
                di_plus=di_plus,
                di_minus=di_minus
            )
        except Exception as e:
            print(f"Failed to calculate enhanced technical indicators: {e}")
            return TechnicalIndicators()
    
    def calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            return macd_line.iloc[-1], macd_signal.iloc[-1], macd_histogram.iloc[-1]
        except:
            return None, None, None
    
    def calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]
        except:
            return None, None, None
    
    def calculate_obv(self, df: pd.DataFrame):
        """Calculate On-Balance Volume (OBV)"""
        try:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            return obv[-1]
        except:
            return None
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period=14, d_period=3):
        """Calculate Stochastic indicator"""
        try:
            lowest_low = df['low'].rolling(k_period).min()
            highest_high = df['high'].rolling(k_period).max()
            k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
            d_percent = k_percent.rolling(d_period).mean()
            return k_percent.iloc[-1], d_percent.iloc[-1]
        except:
            return None, None
    
    def calculate_williams_r(self, df: pd.DataFrame, period=14):
        """Calculate Williams %R"""
        try:
            highest_high = df['high'].rolling(period).max()
            lowest_low = df['low'].rolling(period).min()
            williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
            return williams_r.iloc[-1]
        except:
            return None
    
    def calculate_cci(self, df: pd.DataFrame, period=20):
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(period).mean()
            mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci.iloc[-1]
        except:
            return None
    
    def calculate_adx(self, df: pd.DataFrame, period=14):
        """Calculate Average Directional Index"""
        try:
            high_diff = df['high'].diff()
            low_diff = df['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_dm = pd.Series(plus_dm).rolling(period).mean()
            minus_dm = pd.Series(minus_dm).rolling(period).mean()
            
            tr = self.calculate_true_range(df)
            atr = tr.rolling(period).mean()
            
            plus_di = 100 * (plus_dm / atr)
            minus_di = 100 * (minus_dm / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
        except:
            return None, None, None
    
    def calculate_true_range(self, df: pd.DataFrame):
        """Calculate true range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            return pd.Series(tr)
        except:
            return pd.Series([0] * len(df))
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1] if not atr.empty else 0.0
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return 0.0
    
    def calculate_moving_averages(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Moving averages"""
        ma_short = prices.rolling(window=self.ma_short).mean()
        ma_long = prices.rolling(window=self.ma_long).mean()
        return ma_short, ma_long
    
    def calculate_bollinger_bands_series(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger BandsSeries"""
        ma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        upper = ma + (std * self.bb_std)
        lower = ma - (std * self.bb_std)
        return upper, ma, lower
    
    def calculate_volume_ratio(self, volumes: pd.Series) -> pd.Series:
        """Calculate volume ratio"""
        avg_volume = volumes.rolling(window=20).mean()
        return volumes / avg_volume
    
    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze technical indicators"""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Calculate technical indicators
        rsi = self.calculate_rsi(close, self.rsi_period)
        ma_short, ma_long = self.calculate_moving_averages(close)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        volume_ratio = self.calculate_volume_ratio(volume)
        
        # Get latest values
        latest_rsi = rsi.iloc[-1]
        latest_ma_short = ma_short.iloc[-1]
        latest_ma_long = ma_long.iloc[-1]
        latest_price = close.iloc[-1]
        latest_bb_upper = bb_upper.iloc[-1]
        latest_bb_lower = bb_lower.iloc[-1]
        latest_volume_ratio = volume_ratio.iloc[-1]
        
        # Price trend
        price_trend = "up" if latest_ma_short > latest_ma_long else "down"
        
        # RSI signal
        rsi_signal = "oversold" if latest_rsi < 30 else "overbought" if latest_rsi > 70 else "neutral"
        
        # Bollinger Bands
        bb_signal = "upper" if latest_price > latest_bb_upper else "lower" if latest_price < latest_bb_lower else "middle"
        
        # Volume signal
        volume_signal = "high" if latest_volume_ratio > 1.5 else "low" if latest_volume_ratio < 0.5 else "normal"
        
        return {
            'rsi': latest_rsi,
            'rsi_signal': rsi_signal,
            'ma_short': latest_ma_short,
            'ma_long': latest_ma_long,
            'price_trend': price_trend,
            'bb_upper': latest_bb_upper,
            'bb_lower': latest_bb_lower,
            'bb_signal': bb_signal,
            'volume_ratio': latest_volume_ratio,
            'volume_signal': volume_signal,
            'current_price': latest_price
        }
    
    def generate_trading_signal(self, symbol: str) -> Dict:
        """"""
        try:
            # K
            df = self.get_kline_data(symbol, '3m', 100)
            if df.empty:
                return {'signal': 'hold', 'reason': 'Insufficient data'}
            
            # Analyze technical indicators
            indicators = self.analyze_technical_indicators(df)
            if not indicators:
                return {'signal': 'hold', 'reason': 'Indicator calculation failed'}
            
            # AI
            signal_score = 0
            reasons = []
            
            # RSI signal
            if indicators['rsi_signal'] == 'oversold':
                signal_score += 2
                reasons.append('RSI')
            elif indicators['rsi_signal'] == 'overbought':
                signal_score -= 2
                reasons.append('RSI')
            
            # Moving averages
            if indicators['price_trend'] == 'up':
                signal_score += 1
                reasons.append('Price trend')
            else:
                signal_score -= 1
                reasons.append('Price trend')
            
            # Bollinger Bands
            if indicators['bb_signal'] == 'lower':
                signal_score += 1
                reasons.append('Bollinger Bands')
            elif indicators['bb_signal'] == 'upper':
                signal_score -= 1
                reasons.append('Bollinger Bands')
            
            # Volume signal
            if indicators['volume_signal'] == 'high':
                signal_score += 1
                reasons.append('')
            elif indicators['volume_signal'] == 'low':
                signal_score -= 0.5
                reasons.append('')
            
            # Overall assessment
            if signal_score >= 3:
                signal = 'buy'
            elif signal_score <= -3:
                signal = 'sell'
            else:
                signal = 'hold'
            
            return {
                'symbol': symbol,
                'signal': signal,
                'score': signal_score,
                'reasons': reasons,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'signal': 'hold', 'reason': f'Analysis failed: {str(e)}'}
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            balance = self.safe_api_call(self.exchange.fetch_balance, params={'accountType': 'spot'})
            return balance
        except Exception as e:
            print(f": {e}")
            return {}
    
    def calculate_position_size(self, symbol: str, signal: str) -> float:
        """Calculate position size"""
        try:
            balance = self.get_account_balance()
            quote_currency = symbol.split('/')[1]  # USDT
            
            if quote_currency in balance:
                available_balance = balance[quote_currency]['free']
                position_size = available_balance * self.max_position_size
                
                # 
                if position_size >= self.min_order_amount:
                    return position_size
            
            return 0
        except Exception as e:
            print(f": {e}")
            return 0
    
    def execute_trade(self, symbol: str, signal: str, amount: float) -> Dict:
        """Execute trade"""
        try:
            if signal == 'buy':
                result = self.exchange.create_market_buy_order(symbol, amount, {
                    'accountType': 'spot',
                    'instType': 'SPOT'
                })
            elif signal == 'sell':
                result = self.exchange.create_market_sell_order(symbol, amount, {
                    'accountType': 'spot',
                    'instType': 'SPOT'
                })
            else:
                return {'success': False, 'message': ''}
            
            return {
                'success': True,
                'order_id': result['id'],
                'symbol': symbol,
                'side': signal,
                'amount': amount,
                'result': result
            }
        except Exception as e:
            return {'success': False, 'message': f': {str(e)}'}
    
    def run_ai_trading_cycle(self) -> List[Dict]:
        """Run AI trading cycle"""
        results = []
        
        for symbol in self.symbols:
            try:
                # 
                signal_data = self.generate_trading_signal(symbol)
                
                if signal_data['signal'] in ['buy', 'sell']:
                    # Calculate position size
                    position_size = self.calculate_position_size(symbol, signal_data['signal'])
                    
                    if position_size > 0:
                        # Execute trade
                        trade_result = self.execute_trade(symbol, signal_data['signal'], position_size)
                        
                        results.append({
                            'symbol': symbol,
                            'signal_data': signal_data,
                            'position_size': position_size,
                            'trade_result': trade_result
                        })
                        
                        print(f"AI: {symbol} {signal_data['signal']} {position_size} USDT")
                    else:
                        results.append({
                            'symbol': symbol,
                            'signal_data': signal_data,
                            'position_size': 0,
                            'trade_result': {'success': False, 'message': ''}
                        })
                else:
                    results.append({
                        'symbol': symbol,
                        'signal_data': signal_data,
                        'position_size': 0,
                        'trade_result': {'success': False, 'message': ''}
                    })
                    
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'signal_data': {'signal': 'hold', 'reason': f': {str(e)}'},
                    'position_size': 0,
                    'trade_result': {'success': False, 'message': str(e)}
                })
        
        return results
    
    def set_api_keys(self, openai_key: str = None, gemini_key: str = None, deepseek_key: str = None):
        """Set LLM API keys"""
        if openai_key:
            self.openai_api_key = openai_key
            # API
        if gemini_key:
            self.gemini_api_key = gemini_key
            genai.configure(api_key=gemini_key)
        if deepseek_key:
            self.deepseek_api_key = deepseek_key
    
    def call_openai_gpt(self, prompt: str) -> Dict:
        """Call OpenAI GPT API"""
        try:
            if not self.openai_api_key:
                return {'action': 'hold', 'reason': 'OpenAI API Key'}
            
            # OpenAI API
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "AIJSON"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # AIJSON
            ai_response = response.choices[0].message.content.strip()
            try:
                decision = json.loads(ai_response)
                return decision
            except json.JSONDecodeError:
                # AIJSONJSON
                import re
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    return decision
                else:
                    return {'action': 'hold', 'reason': f'AI: {ai_response}'}
                    
        except Exception as e:
            return {'action': 'hold', 'reason': f'OpenAI API: {str(e)}'}
    
    def call_google_gemini(self, prompt: str) -> Dict:
        """Call Google Gemini API (latest version)"""
        try:
            if not self.gemini_api_key:
                print(" Gemini API Key")
                return self.simulate_global_decision({})
            
            # Google Gemini API
            from google import genai
            
            client = genai.Client(api_key=self.gemini_api_key)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # Gemini 2.5 Flash
                contents=f"AIJSONmarkdown\n\n{prompt}",
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 2000,  # token
                }
            )
            
            # AIJSON
            ai_response = response.text.strip()
            print(f" Gemini: {ai_response[:300]}...")
            
            # markdown
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            ai_response = ai_response.strip()
            
            print(f" : {ai_response[:300]}...")
            
            try:
                decision = json.loads(ai_response)
                print(f" JSON")
                return decision
            except json.JSONDecodeError as e:
                print(f" JSON: {e}")
                # AIJSONJSON
                import re
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group())
                        print(f" JSON")
                        return decision
                    except json.JSONDecodeError as e2:
                        print(f" JSON: {e2}")
                        print(f": {ai_response}")
                        return self.simulate_global_decision({})
                else:
                    print(f" JSON")
                    print(f": {ai_response}")
                    return self.simulate_global_decision({})
                    
        except Exception as e:
            print(f" Gemini API: {str(e)}")
            return self.simulate_global_decision({})
    
    def call_deepseek_v3(self, prompt: str) -> Dict:
        """Call DeepSeek V3 API"""
        try:
            if not self.deepseek_api_key:
                print(" DeepSeek API Key")
                return self.simulate_global_decision({})
            
            # OpenAIAPIDeepSeek
            client = openai.OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "AIJSONmarkdown"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            
            # AIJSON
            ai_response = response.choices[0].message.content.strip()
            print(f" DeepSeek: {ai_response[:300]}...")
            
            # markdown
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            # 
            ai_response = ai_response.strip()
            print(f" : {ai_response[:300]}...")
            
            try:
                # JSON
                decision = json.loads(ai_response)
                print(f" JSON")
                return decision
            except json.JSONDecodeError as e:
                print(f" JSON: {e}")
                print(f"LLM: {ai_response}")
                
                # JSON
                import re
                try:
                    json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        decision = json.loads(json_str)
                        print(f" JSON")
                        return decision
                    else:
                        print(f" JSON: {e}")
                        print(f": {ai_response}")
                        return self.simulate_global_decision({})
                except Exception as e2:
                    print(f" JSON: {e2}")
                    print(f": {ai_response}")
                    return self.simulate_global_decision({})
                
        except Exception as e:
            print(f" DeepSeek API: {str(e)}")
            return self.simulate_global_decision({})
    
    def call_llm_for_decision(self, input_data: LLMTradingInput) -> LLMTradingOutput:
        """Call LLM for new format trading decisions"""
        try:
            system_prompt = self.get_system_prompt()
            developer_prompt = self.get_developer_prompt()
            
            # 
            full_prompt = f"{system_prompt}\n\n{developer_prompt}\n\nInput data:\n{input_data.json()}\n\nOutput (JSON only):"
            
            if self.selected_llm == 'gpt' and self.openai_api_key:
                openai.api_key = self.openai_api_key
                response = openai.ChatCompletion.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{developer_prompt}\n\nInput data:\n{input_data.json()}\n\nOutput (JSON only):"}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                llm_response = response.choices[0].message.content.strip()
                
            elif self.selected_llm == 'gemini' and self.gemini_api_key:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                llm_response = response.text.strip()
                
            elif self.selected_llm == 'deepseek' and self.deepseek_api_key:
                client = openai.OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{developer_prompt}\n\nInput data:\n{input_data.json()}\n\nOutput (JSON only):"}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                llm_response = response.choices[0].message.content.strip()
            else:
                # API
                return self._simulate_llm_response(input_data)
            
            try:
                # markdown
                if llm_response.startswith('```json'):
                    llm_response = llm_response[7:]
                if llm_response.startswith('```'):
                    llm_response = llm_response[3:]
                if llm_response.endswith('```'):
                    llm_response = llm_response[:-3]
                llm_response = llm_response.strip()
                
                response_data = json.loads(llm_response)
                return LLMTradingOutput(**response_data)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"LLM response: {llm_response}")
                return self._simulate_llm_response(input_data)
                
        except Exception as e:
            print(f"LLM call failed: {e}")
            # API
            return self._simulate_llm_response(input_data)
    
    def get_system_prompt(self) -> str:
        """Get system prompt"""
        return """You are a "spot scalping execution auditor" for 3-minute crypto trading. You must follow these iron rules:

1. NEVER invent prices, indicators, or facts
2. Only use the numeric fields provided in the input
3. Output ONLY valid JSON, no explanations
4. Gate checks: If ANY condition fails, return FLAT
5. Position management: Move stop to breakeven after TP1, exit after max hold time
6. You can BUY to enter or add positions
7. You can SELL to exit or reduce positions at any time if market conditions deteriorate"""

    def get_developer_prompt(self) -> str:
        """Get developer prompt"""
        return """Input/Output Contract:
- Input: Real-time market data with pre-calculated indicators
- Output: JSON with action (BUY/SELL/FLAT/HOLD/EXIT), entry, stop_loss, take_profit, checklist, confidence, reason_codes

Decision Process:
1. If has_position=true and market conditions deteriorated (trend reversed, stop loss hit risk, etc.) → SELL to exit
2. If has_position=false, check all gating conditions (trend_align, vol_ok, liquidity_ok, rr_ok, risk_limits_ok, no news_risk)
3. If ALL pass → BUY with confidence 80-95%
4. If ANY fail → FLAT with confidence 0-20%
5. Use provided entry price, stop loss, and take profit levels
6. SELL action will close the entire position immediately"""

    def _simulate_llm_response(self, input_data: LLMTradingInput) -> LLMTradingOutput:
        """Simulate LLM response (for testing)"""
        # 
        precalc = input_data.precalc
        
        if not all([
            precalc.trend_align,
            precalc.vol_ok,
            precalc.liquidity_ok,
            precalc.rr_ok,
            precalc.risk_limits_ok,
            not input_data.risk_rules.news_risk,
            not input_data.risk_rules.cooldown_active,
            not input_data.risk_rules.day_max_drawdown_hit
        ]):
            # FLAT
            reason_codes = []
            if not precalc.trend_align:
                reason_codes.append("TREND_FAIL")
            if not precalc.vol_ok:
                reason_codes.append("VOL_FAIL")
            if not precalc.liquidity_ok:
                reason_codes.append("LIQ_FAIL")
            if not precalc.rr_ok:
                reason_codes.append("RR_FAIL")
            if not precalc.risk_limits_ok:
                reason_codes.append("RISK_FAIL")
            if input_data.risk_rules.news_risk:
                reason_codes.append("NEWS")
            if input_data.risk_rules.cooldown_active:
                reason_codes.append("COOLDOWN")
            
            return self._create_flat_response(reason_codes)
        
        # BUY
        entry_price = input_data.prices.candidate_entry or input_data.prices.last_price
        stop_price = input_data.precalc.chosen_stop
        tp_prices = input_data.precalc.tp_prices or []
        
        take_profit = []
        for i, (tp_price, r_mult) in enumerate(zip(tp_prices, input_data.risk_rules.tp_policy)):
            take_profit.append(TakeProfit(
                price=tp_price,
                size_pct=input_data.risk_rules.tp_size_pct[i],
                r_mult=r_mult
            ))
        
        return LLMTradingOutput(
            action="BUY",
            entry=Entry(type="limit", price=entry_price),
            stop_loss=StopLoss(price=stop_price, method="swing_or_atr", atr_mult=input_data.risk_rules.atr_mult),
            take_profit=take_profit,
            time_in_force="GTC",
            max_hold_minutes=input_data.risk_rules.max_hold_minutes,
            checklist=Checklist(
                trend_align=precalc.trend_align,
                vol_ok=precalc.vol_ok,
                liquidity_ok=precalc.liquidity_ok,
                rr_ok=precalc.rr_ok,
                risk_limits_ok=precalc.risk_limits_ok,
                news_risk=input_data.risk_rules.news_risk
            ),
            confidence=85,  # 
            reason_codes=["OK"],
            notes=""
        )
    
    def _create_flat_response(self, reason_codes: List[str]) -> LLMTradingOutput:
        """Create FLAT response"""
        return LLMTradingOutput(
            action="FLAT",
            entry=Entry(type="market", price=None),
            stop_loss=StopLoss(price=None, method="swing_or_atr", atr_mult=1.2),
            take_profit=[],
            time_in_force="GTC",
            max_hold_minutes=30,
            checklist=Checklist(
                trend_align=False,
                vol_ok=False,
                liquidity_ok=True,
                rr_ok=False,
                risk_limits_ok=True,
                news_risk=False
            ),
            confidence=0,
            reason_codes=reason_codes,
            notes=""
        )
    
    def get_current_positions(self) -> Dict:
        """Get current positions (includes full position info)"""
        try:
            balance = self.safe_api_call(self.exchange.fetch_balance, params={'accountType': 'spot'})
            positions = {}
            
            for symbol in self.symbols:
                base_currency = symbol.split('/')[0]
                if base_currency in balance and balance[base_currency]['free'] > 0:
                    current_price = self.get_current_price(symbol)
                    amount = balance[base_currency]['free']
                    value_usdt = amount * current_price
                    
                    # 
                    time.sleep(0.1)  # 100ms
                    symbol_orders = self.get_all_open_orders().get(symbol, [])
                    
                    # 
                    stop_loss_price = None
                    take_profit_prices = []
                    
                    for order in symbol_orders:
                        if order.get('type') == 'stop_loss':
                            stop_loss_price = order.get('trigger_price')
                        elif order.get('type') == 'take_profit':
                            take_profit_prices.append(order.get('price'))
                    
                    positions[symbol] = {
                        'amount': amount,
                        'value_usdt': value_usdt,
                        'current_price': current_price,
                        'entry_price': None,  # 
                        'stop_loss_price': stop_loss_price,
                        'take_profit_prices': take_profit_prices,
                        'unrealized_pnl': 0,  # 
                        'open_orders': symbol_orders,
                        'has_stop_loss': stop_loss_price is not None,
                        'has_take_profit': len(take_profit_prices) > 0
                    }
            
            return positions
        except Exception as e:
            print(f": {e}")
            return {}
    
    def get_all_open_orders(self) -> Dict:
        """Get all open orders (including manual and algo orders)"""
        try:
            all_orders = {}
            
            # 1. 
            try:
                open_orders = self.safe_api_call(self.exchange.fetch_open_orders, params={'accountType': 'spot'})
                for order in open_orders:
                    symbol = order['symbol']
                    if symbol not in all_orders:
                        all_orders[symbol] = []
                    
                    all_orders[symbol].append({
                        'order_id': order['id'],
                        'type': 'limit_order',
                        'side': order['side'],
                        'amount': order['amount'],
                        'price': order['price'],
                        'status': order['status'],
                        'timestamp': order['timestamp'],
                        'filled': order['filled'],
                        'remaining': order['remaining']
                    })
            except Exception as e:
                print(f": {e}")
            
            # 2. /
            try:
                # OKX API - 
                time.sleep(0.2)  # 200ms
                algo_response = self.safe_api_call(
                    self.exchange.private_get_trade_orders_algo_pending,
                    {'instType': 'SPOT', 'ordType': 'conditional'}
                )
                
                if algo_response and 'data' in algo_response:
                    for algo_order in algo_response['data']:
                        symbol = algo_order.get('instId', '').replace('-', '/')
                        if symbol and '/' in symbol:
                            if symbol not in all_orders:
                                all_orders[symbol] = []
                            
                            # 
                            order_type = 'take_profit' if algo_order.get('side') == 'sell' else 'stop_loss'
                            
                            # 
                            def safe_float(value, default=0.0):
                                if value is None or value == '' or value == 'None':
                                    return default
                                try:
                                    return float(value)
                                except (ValueError, TypeError):
                                    return default
                            
                            def safe_int(value, default=0):
                                if value is None or value == '' or value == 'None':
                                    return default
                                try:
                                    return int(value)
                                except (ValueError, TypeError):
                                    return default
                            
                            all_orders[symbol].append({
                                'order_id': algo_order.get('algoId'),
                                'type': order_type,
                                'side': algo_order.get('side'),
                                'amount': safe_float(algo_order.get('sz', 0)),
                                'price': safe_float(algo_order.get('ordPx', 0)),
                                'trigger_price': safe_float(algo_order.get('triggerPx', 0)),
                                'status': algo_order.get('state'),
                                'timestamp': safe_int(algo_order.get('cTime', 0)),
                                'algo_order': True
                            })
            except Exception as e:
                print(f": {e}")
            
            # 3. trigger, oco, move_order_stop
            try:
                for order_type in ['trigger', 'oco', 'move_order_stop']:
                    time.sleep(0.3)  # 300ms
                    algo_response = self.safe_api_call(
                        self.exchange.private_get_trade_orders_algo_pending,
                        {'instType': 'SPOT', 'ordType': order_type}
                    )
                    
                    if algo_response and 'data' in algo_response:
                        for algo_order in algo_response['data']:
                            symbol = algo_order.get('instId', '').replace('-', '/')
                            if symbol and '/' in symbol:
                                if symbol not in all_orders:
                                    all_orders[symbol] = []
                                
                                all_orders[symbol].append({
                                    'order_id': algo_order.get('algoId'),
                                    'type': order_type,
                                    'side': algo_order.get('side'),
                                    'amount': float(algo_order.get('sz', 0)),
                                    'price': float(algo_order.get('ordPx', 0)) if algo_order.get('ordPx') else 0,
                                    'trigger_price': float(algo_order.get('triggerPx', 0)) if algo_order.get('triggerPx') else 0,
                                    'status': algo_order.get('state'),
                                    'timestamp': int(algo_order.get('cTime', 0)),
                                    'algo_order': True
                                })
            except Exception as e:
                print(f": {e}")
            
            # 
            # try:
            #     # OKX
            #     algo_orders = self.exchange.private_get_trade_order_algo({
            #         'instType': 'SPOT',
            #         'state': 'effective'  # 
            #     })
            #     
            #     if algo_orders and 'data' in algo_orders and len(algo_orders['data']) > 0:
            #         for algo_order in algo_orders['data']:
            #             symbol = algo_order['instId']
            #             if symbol not in all_orders:
            #                 all_orders[symbol] = []
            #             
            #             # 
            #             order_type = 'take_profit_stop_loss'
            #             if algo_order.get('tpTriggerPx') and algo_order.get('slTriggerPx'):
            #                 order_type = 'take_profit_stop_loss'
            #             elif algo_order.get('tpTriggerPx'):
            #                 order_type = 'take_profit'
            #             elif algo_order.get('slTriggerPx'):
            #                 order_type = 'stop_loss'
            #             
            #             all_orders[symbol].append({
            #                 'order_id': algo_order['algoId'],
            #                 'type': order_type,
            #                 'side': algo_order['side'],
            #                 'amount': float(algo_order['sz']),
            #                 'price': float(algo_order.get('tpTriggerPx', algo_order.get('slTriggerPx', 0))),
            #                 'trigger_price': float(algo_order.get('tpTriggerPx', algo_order.get('slTriggerPx', 0))),
            #                 'status': algo_order['state'],
            #                 'timestamp': int(algo_order['cTime']),
            #                 'algo_order': True
            #             })
            #     else:
            #         # 
            #         pass
            # except Exception as e:
            #     # 
            #     print(f": {e}")
            #     pass
            
            # pending_orders
            pass
            
            return all_orders
        except Exception as e:
            print(f": {e}")
            return {}
    
    def get_recent_trades(self) -> Dict:
        """Get recent trades"""
        try:
            recent_trades = {}
            
            for symbol in self.symbols:
                try:
                    # Get recent trades
                    trades = self.safe_api_call(self.exchange.fetch_my_trades, symbol, limit=10, params={'accountType': 'spot'})
                    
                    if trades:
                        recent_trades[symbol] = []
                        for trade in trades[-5:]:  # 5
                            recent_trades[symbol].append({
                                'trade_id': trade['id'],
                                'order_id': trade['order'],
                                'side': trade['side'],
                                'amount': trade['amount'],
                                'price': trade['price'],
                                'cost': trade['cost'],
                                'fee': trade['fee'],
                                'timestamp': trade['timestamp'],
                                'datetime': trade['datetime']
                            })
                except Exception as e:
                    print(f" {symbol} : {e}")
                    continue
            
            return recent_trades
        except Exception as e:
            print(f": {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price"""
        try:
            ticker = self.safe_api_call(self.exchange.fetch_ticker, symbol)
            return ticker['last']
        except Exception as e:
            print(f" {symbol}: {e}")
            return 0
    
    def get_market_data_for_llm(self, symbol: str):
        """Prepare market data for LLM (backward compatible)"""
        return self.get_enhanced_market_data_for_llm(symbol)
    
    def get_enhanced_market_data_for_llm(self, symbol: str) -> EnhancedLLMTradingInput:
        """Prepare enhanced market data for LLM"""
        try:
            # K
            df = self.get_kline_data(symbol, '3m', 200)  # 
            if df.empty:
                return None
            
            # Get current price24
            ticker = self.safe_api_call(self.exchange.fetch_ticker, symbol)
            if not ticker:
                print(f"{symbol}ticker")
                return None
            current_price = ticker['last']
            
            # 
            technical_indicators = self.calculate_enhanced_technical_indicators(df)
            
            # Calculate market structure
            market_structure = self.calculate_market_structure(df)
            
            # Get order book data
            order_book = self.get_order_book_data(symbol)
            
            # Get market sentiment data
            market_sentiment = self.get_market_sentiment(symbol)
            
            # Get time-based data
            time_data = self.get_time_based_data(symbol, df)
            
            # 
            positions = self.get_current_positions()
            current_position = positions.get(symbol, {})
            position_amount = current_position.get('amount', 0)
            
            # 
            symbol_orders = self.get_all_open_orders().get(symbol, [])
            
            # 
            atr = self.calculate_atr(df, 14)
            atr_pct = (atr / current_price) * 100
            
            # 
            enhanced_input = EnhancedLLMTradingInput(
                symbol=symbol,
                timeframe="3m",
                now_utc=datetime.now().isoformat(),
                
                # 
                prices=EnhancedPrices(
                    last_price=current_price,
                    candidate_entry=current_price,
                    bid=ticker.get('bid'),
                    ask=ticker.get('ask'),
                    open_24h=ticker.get('open'),
                    high_24h=ticker.get('high'),
                    low_24h=ticker.get('low'),
                    volume_24h=ticker.get('baseVolume'),
                    change_24h=ticker.get('change'),
                    change_percent_24h=ticker.get('percentage')
                ),
                
                # 
                technical_indicators=technical_indicators,
                
                # 
                market_structure=market_structure,
                
                # 
                order_book=order_book,
                
                # 
                market_sentiment=market_sentiment,
                
                # 
                time_data=time_data,
                
                # 
                trend_filters=TrendFilters(
                    m15_trend="bullish" if technical_indicators.ema_20 and technical_indicators.ema_50 and technical_indicators.ema_20 > technical_indicators.ema_50 else "bearish",
                    h1_trend="bullish" if technical_indicators.ema_20 and technical_indicators.ema_50 and technical_indicators.ema_20 > technical_indicators.ema_50 else "bearish",
                    ema20_gt_ema50_m15=technical_indicators.ema_20 and technical_indicators.ema_50 and technical_indicators.ema_20 > technical_indicators.ema_50,
                    ema20_slope_3m=float((technical_indicators.ema_20 - technical_indicators.ema_20) / technical_indicators.ema_20 * 100) if technical_indicators.ema_20 else 0
                ),
                
                volatility=Volatility(
                    atr_3m_14=atr,
                    atr_pct=atr_pct,
                    vol_state="ok" if 0.5 <= atr_pct <= 3.0 else "high" if atr_pct > 3.0 else "low",
                    min_atr_pct=0.5,
                    max_atr_pct=3.0
                ),
                
                structure_signals=StructureSignals(
                    recent_swing_low=market_structure.recent_swing_low or float(df['low'].tail(10).min()),
                    recent_swing_high=market_structure.recent_swing_high or float(df['high'].tail(10).max()),
                    broke_recent_high_n=False,
                    vwap_distance_bp=0.0,
                    volume_vs20x=technical_indicators.volume_ratio or 1.0
                ),
                
                liquidity_costs=LiquidityCosts(
                    spread_bp=2.0,
                    est_slippage_bp=1.0,
                    taker_fee_bp=5.0,
                    maker_fee_bp=3.0,
                    costs_ok=True,
                    cost_share_of_edge=0.05
                ),
                
                risk_rules=RiskRules(
                    account_equity=1000.0,
                    max_risk_pct=0.5,
                    min_rr=1.2,
                    day_max_drawdown_hit=False,
                    cooldown_active=False,
                    news_risk=market_sentiment.news_sentiment == "negative",
                    atr_mult=1.2,
                    breakeven_after_tp1=True,
                    tp_policy=[1.2, 2.0],
                    tp_size_pct=[50, 50],
                    max_hold_minutes=30
                ),
                
                precalc=Precalc(
                    swing_stop=market_structure.recent_swing_low * 0.98 if market_structure.recent_swing_low else current_price * 0.98,
                    atr_stop=current_price - (atr * 1.5),
                    chosen_stop=current_price * 0.98,
                    r_per_dollar=1.2,
                    tp_prices=[current_price * 1.024, current_price * 1.04],
                    rr_ok=True,
                    trend_align=True,
                    vol_ok=0.5 <= atr_pct <= 3.0,
                    liquidity_ok=True,
                    risk_limits_ok=True
                ),
                
                position_state=PositionState(
                    has_position=position_amount > 0,
                    entry_price=current_position.get('entry_price'),
                    stop_loss=current_position.get('stop_loss_price'),
                    tp_levels=current_position.get('take_profit_prices', []),
                    tp_filled=[False] * len(current_position.get('take_profit_prices', [])),
                    remaining_size_pct=100.0 if position_amount > 0 else 0.0,
                    minutes_in_position=None,  # 
                    # 
                    current_amount=position_amount,
                    current_value_usdt=current_position.get('value_usdt', 0),
                    unrealized_pnl=current_position.get('unrealized_pnl', 0),
                    has_stop_loss=current_position.get('has_stop_loss', False),
                    has_take_profit=current_position.get('has_take_profit', False),
                    open_orders_count=len(symbol_orders)
                )
            )
            
            return enhanced_input
            
        except Exception as e:
            print(f"Failed to prepare enhanced market data {symbol}: {e}")
            return None
    
    def calculate_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Calculate market structure"""
        try:
            # 
            highs = df['high'].rolling(20).max()
            lows = df['low'].rolling(20).min()
            
            # 
            support_levels = []
            resistance_levels = []
            
            # 
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            pivot_point = (high + low + close) / 3
            pivot_resistance_1 = 2 * pivot_point - low
            pivot_resistance_2 = pivot_point + (high - low)
            pivot_support_1 = 2 * pivot_point - high
            pivot_support_2 = pivot_point - (high - low)
            
            # 
            recent_highs = df['high'].tail(10)
            recent_lows = df['low'].tail(10)
            
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-3]
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-3]
            lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-3]
            lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-3]
            
            # 
            if higher_highs and higher_lows:
                trend_strength = "strong_up"
            elif higher_highs:
                trend_strength = "up"
            elif lower_highs and lower_lows:
                trend_strength = "strong_down"
            elif lower_lows:
                trend_strength = "down"
            else:
                trend_strength = "sideways"
            
            return MarketStructure(
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                pivot_point=pivot_point,
                pivot_resistance_1=pivot_resistance_1,
                pivot_resistance_2=pivot_resistance_2,
                pivot_support_1=pivot_support_1,
                pivot_support_2=pivot_support_2,
                higher_highs=higher_highs,
                higher_lows=higher_lows,
                lower_highs=lower_highs,
                lower_lows=lower_lows,
                trend_strength=trend_strength,
                recent_swing_low=float(df['low'].tail(20).min()),
                recent_swing_high=float(df['high'].tail(20).max()),
                swing_low_time=None,
                swing_high_time=None
            )
        except Exception as e:
            print(f"Calculate market structure: {e}")
            return MarketStructure()
    
    def get_order_book_data(self, symbol: str) -> OrderBookData:
        """Get order book data"""
        try:
            order_book = self.safe_api_call(self.exchange.fetch_order_book, symbol, limit=10)
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            # 
            bid_depth_1 = sum([bid[1] for bid in bids[:1]])
            bid_depth_5 = sum([bid[1] for bid in bids[:5]])
            bid_depth_10 = sum([bid[1] for bid in bids[:10]])
            
            ask_depth_1 = sum([ask[1] for ask in asks[:1]])
            ask_depth_5 = sum([ask[1] for ask in asks[:5]])
            ask_depth_10 = sum([ask[1] for ask in asks[:10]])
            
            # 
            bid_ask_ratio = bid_depth_10 / ask_depth_10 if ask_depth_10 > 0 else 1.0
            
            # 
            order_book_imbalance = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10) if (bid_depth_10 + ask_depth_10) > 0 else 0
            
            # 
            large_bid_orders = len([bid for bid in bids if bid[0] * bid[1] > 10000])
            large_ask_orders = len([ask for ask in asks if ask[0] * ask[1] > 10000])
            
            return OrderBookData(
                bid_depth_1=bid_depth_1,
                bid_depth_5=bid_depth_5,
                bid_depth_10=bid_depth_10,
                ask_depth_1=ask_depth_1,
                ask_depth_5=ask_depth_5,
                ask_depth_10=ask_depth_10,
                bid_ask_ratio=bid_ask_ratio,
                order_book_imbalance=order_book_imbalance,
                large_bid_orders=large_bid_orders,
                large_ask_orders=large_ask_orders,
                large_order_threshold=10000.0
            )
        except Exception as e:
            print(f"Get order book data: {e}")
            return OrderBookData()
    
    def get_market_sentiment(self, symbol: str) -> MarketSentiment:
        """Get market sentiment data"""
        try:
            # APIGet market sentiment data
            # 
            return MarketSentiment(
                fear_greed_index=50.0,  # 
                social_sentiment="neutral",
                news_sentiment="neutral",
                money_flow_index=None,
                accumulation_distribution=None,
                active_addresses=None,
                transaction_count=None,
                network_value=None
            )
        except Exception as e:
            print(f"Get market sentiment data: {e}")
            return MarketSentiment()
    
    def get_time_based_data(self, symbol: str, df: pd.DataFrame) -> TimeBasedData:
        """Get time-based data"""
        try:
            now = datetime.now()
            
            # 
            hour = now.hour
            if 0 <= hour < 8:
                market_session = "asian"
            elif 8 <= hour < 16:
                market_session = "european"
            elif 16 <= hour < 24:
                market_session = "american"
            else:
                market_session = "overlap"
            
            # 
            volatility_by_hour = {}
            volume_by_hour = {}
            
            for i in range(24):
                hour_data = df[df.index.hour == i]
                if not hour_data.empty:
                    volatility_by_hour[str(i)] = float(hour_data['close'].std())
                    volume_by_hour[str(i)] = float(hour_data['volume'].mean())
            
            return TimeBasedData(
                current_time=now.isoformat(),
                market_session=market_session,
                volatility_by_hour=volatility_by_hour,
                volume_by_hour=volume_by_hour,
                same_time_yesterday=None,
                same_time_last_week=None,
                same_time_last_month=None,
                day_of_week=now.weekday(),
                hour_of_day=hour,
                is_weekend=now.weekday() >= 5,
                is_holiday=False
            )
        except Exception as e:
            print(f": {e}")
            return TimeBasedData(
                current_time=datetime.now().isoformat(),
                market_session="unknown",
                day_of_week=datetime.now().weekday(),
                hour_of_day=datetime.now().hour
            )
        """LLM - LLM"""
        try:
            # K
            df = self.get_kline_data(symbol, '3m', 100)
            if df.empty:
                return None
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                return None
            
            # Calculate technical indicators
            atr = self.calculate_atr(df, 14)
            atr_pct = (atr / current_price) * 100
            
            # EMA
            df['ema20'] = df['close'].ewm(span=20).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()
            
            # 
            positions = self.get_current_positions()
            current_position = positions.get(symbol, {})
            position_amount = current_position.get('amount', 0)
            
            # 
            input_data = LLMTradingInput(
                symbol=symbol,
                timeframe="3m",
                now_utc=datetime.now().isoformat(),
                prices=Prices(
                    last_price=current_price,
                    candidate_entry=current_price,
                    bid=current_price * 0.9999,  # 
                    ask=current_price * 1.0001
                ),
                trend_filters=TrendFilters(
                    m15_trend="bullish" if df['ema20'].iloc[-1] > df['ema50'].iloc[-1] else "bearish",
                    h1_trend="bullish" if df['ema20'].iloc[-1] > df['ema50'].iloc[-1] else "bearish",
                    ema20_gt_ema50_m15=df['ema20'].iloc[-1] > df['ema50'].iloc[-1],
                    ema20_slope_3m=float((df['ema20'].iloc[-1] - df['ema20'].iloc[-5]) / df['ema20'].iloc[-5] * 100)
                ),
                volatility=Volatility(
                    atr_3m_14=atr,
                    atr_pct=atr_pct,
                    vol_state="ok" if 0.5 <= atr_pct <= 3.0 else "high" if atr_pct > 3.0 else "low",
                    min_atr_pct=0.5,
                    max_atr_pct=3.0
                ),
                structure_signals=StructureSignals(
                    recent_swing_low=float(df['low'].tail(10).min()),
                    recent_swing_high=float(df['high'].tail(10).max()),
                    broke_recent_high_n=False,  # 
                    vwap_distance_bp=0.0,  # 
                    volume_vs20x=1.0  # 
                ),
                liquidity_costs=LiquidityCosts(
                    spread_bp=2.0,  # 
                    est_slippage_bp=1.0,
                    taker_fee_bp=5.0,
                    maker_fee_bp=3.0,
                    costs_ok=True,
                    cost_share_of_edge=0.05
                ),
                risk_rules=RiskRules(
                    account_equity=1000.0,  # 
                    max_risk_pct=0.5,
                    min_rr=1.2,
                    day_max_drawdown_hit=False,
                    cooldown_active=False,
                    news_risk=False,
                    atr_mult=1.2,
                    breakeven_after_tp1=True,
                    tp_policy=[1.2, 2.0],
                    tp_size_pct=[50, 50],
                    max_hold_minutes=30
                ),
                precalc=Precalc(
                    swing_stop=float(df['low'].tail(10).min() * 0.98),  # 
                    atr_stop=current_price - (atr * 1.5),
                    chosen_stop=current_price * 0.98,  # 2%
                    r_per_dollar=1.2,
                    tp_prices=[current_price * 1.024, current_price * 1.04],  # 
                    rr_ok=True,
                    trend_align=True,  # 
                    vol_ok=0.5 <= atr_pct <= 3.0,
                    liquidity_ok=True,
                    risk_limits_ok=True
                ),
                position_state=PositionState(
                    has_position=position_amount > 0,
                    entry_price=None,
                    stop_loss=None,
                    tp_levels=None,
                    tp_filled=None,
                    remaining_size_pct=None,
                    minutes_in_position=None
                )
            )
            
            return input_data
            
        except Exception as e:
            print(f"Failed to prepare market data {symbol}: {e}")
            return None
    
    def call_llm_for_global_decision(self, all_market_data: Dict) -> Dict:
        """Call LLM for global trading decisions (analyze all symbols at once)"""
        try:
            # 
            prompt = self.build_global_trading_prompt(all_market_data)
            
            # LLM API
            if self.selected_llm == 'gpt':
                if self.openai_api_key:
                    decision = self.call_openai_gpt(prompt)
                else:
                    # 
                    decision = self.simulate_global_decision(all_market_data)
                    decision['reason'] = f"[] {decision['reason']}"
            elif self.selected_llm == 'gemini':
                if self.gemini_api_key:
                    decision = self.call_google_gemini(prompt)
                else:
                    # 
                    decision = self.simulate_global_decision(all_market_data)
                    decision['reason'] = f"[] {decision['reason']}"
            elif self.selected_llm == 'deepseek':
                if self.deepseek_api_key:
                    decision = self.call_deepseek_v3(prompt)
                else:
                    # 
                    decision = self.simulate_global_decision(all_market_data)
                    decision['reason'] = f"[] {decision['reason']}"
            else:
                decision = {'trading_plan': [], 'reason': 'LLM'}
            
            # AI
            decision['prompt'] = prompt
            decision['llm_model'] = self.selected_llm
            decision['timestamp'] = datetime.now().isoformat()
            
            return decision
        except Exception as e:
            return {'trading_plan': [], 'reason': f'LLM: {str(e)}'}
    
    def build_global_trading_prompt(self, all_market_data: Dict) -> str:
        """Build global trading prompt"""
        # 
        try:
            balance = self.exchange.fetch_balance(params={'accountType': 'spot'})
            total_available_usdt = balance.get('USDT', {}).get('free', 0)
        except:
            total_available_usdt = 1000.0
        
        # 
        symbols_data = []
        for symbol, data in all_market_data.items():
            # data  LLMTradingInput 
            symbols_data.append(f"""
{symbol}:
  : {data.prices.last_price} USDT
  : {'' if data.position_state.has_position else ''}
  :
  - 15: {data.trend_filters.m15_trend}
  - 1: {data.trend_filters.h1_trend}
  - EMA20>EMA50: {data.trend_filters.ema20_gt_ema50_m15}
  :
  - ATR: {data.volatility.atr_pct:.2f}%
  - : {data.volatility.vol_state}
  :
  - : {data.risk_rules.max_risk_pct}%
  - : {data.risk_rules.min_rr}
  - : {'' if data.risk_rules.cooldown_active else ''}
""")
        
        symbols_text = "\n".join(symbols_data)
        
        prompt = f"""
AI

:
- USDT: {total_available_usdt:.2f} USDT
- : 200.0 USDT

:
{symbols_text}

:
1. 3
2. ****: 
3. ****: 0.2%
   - >3%4-6%2.5-3.5%
   - 1-3%3-4%2-2.5%
   - <1%2.5-3%1.5-2%
   - 2.5%
4. ****
   - BUY: 
   - SELL:   
   - HOLD: 
5. 
6. 
7. ****""

JSON:
{{
    "total_available_usdt": {total_available_usdt},
    "trading_plan": [
        {{
            "symbol": "BTC/USDT",
            "decision": {{
                "action": "buy/sell/hold",
                "amount_usdt": hold0,
                "target_price": 2.5%hold,
                "stop_loss_price": 1.5-3%hold,
                "reason": "KRSI"
            }}
        }},
        {{
            "symbol": "ETH/USDT",
            "decision": {{...}}
        }},
        ...
    ],
    "reason": ""
}}
"""
        return prompt
    
    def simulate_global_decision(self, all_market_data: Dict) -> Dict:
        """Simulate global decision (for demo)"""
        trading_plan = []
        total_available_usdt = 0
        
        # 
        for symbol, data in all_market_data.items():
            total_available_usdt = max(total_available_usdt, data.get('available_usdt', 0))
        
        # 
        sorted_symbols = sorted(all_market_data.items(), 
                              key=lambda x: x[1].get('kline_data', {}).get('price_volatility_20candles', 0), 
                              reverse=True)
        
        remaining_usdt = total_available_usdt
        
        for symbol, data in sorted_symbols:
            indicators = data['technical_indicators']
            current_price = data['current_price']
            position = data['current_position']
            kline_data = data.get('kline_data', {})
            
            # 
            price_volatility = kline_data.get('price_volatility_20candles', 2.0)
            avg_candle_range = kline_data.get('avg_candle_range_pct', 1.0)
            
            rsi = indicators.get('rsi', 50)
            price_trend = indicators.get('price_trend', 'neutral')
            bb_signal = indicators.get('bb_signal', 'middle')
            volume_signal = indicators.get('volume_signal', 'normal')
            
            # 
            if price_volatility > 3:
                tp_pct = 0.04   # 4%
                sl_pct = 0.025  # 2.5%
            elif price_volatility > 1.5:
                tp_pct = 0.03   # 3%
                sl_pct = 0.02   # 2%
            else:
                tp_pct = 0.025  # 2.5%
                sl_pct = 0.015  # 1.5%
            
            #  - 
            if (rsi < 30 and price_trend == 'up' and bb_signal == 'lower' and 
                volume_signal == 'high' and remaining_usdt >= 50 and price_volatility > 1.0):
                
                # 50%
                amount_usdt = min(remaining_usdt * 0.5, 100.0)
                remaining_usdt -= amount_usdt
                
                decision = {
                    'action': 'buy',
                    'amount_usdt': amount_usdt,
                    'target_price': current_price * (1 + tp_pct),
                    'stop_loss_price': current_price * (1 - sl_pct),
                    'reason': f'[] RSI={rsi:.1f}Price trend={price_trend}Bollinger Bands={bb_signal}={volume_signal}={price_volatility:.1f}%{amount_usdt:.1f} USDT'
                }
            elif (rsi > 70 and price_trend == 'down' and bb_signal == 'upper' and 
                  volume_signal == 'high' and position.get('amount', 0) > 0):
                
                decision = {
                    'action': 'sell',
                    'amount_usdt': position.get('amount', 0) * current_price,
                    'target_price': current_price,
                    'stop_loss_price': current_price,
                    'reason': f'[] RSI={rsi:.1f}Price trend={price_trend}Bollinger Bands={bb_signal}={volume_signal}={price_volatility:.1f}%'
                }
            elif (rsi < 40 and price_trend == 'up' and remaining_usdt >= 30 and price_volatility > 0.8):
                # 
                amount_usdt = min(remaining_usdt * 0.3, 50.0)
                remaining_usdt -= amount_usdt
                
                decision = {
                    'action': 'buy',
                    'amount_usdt': amount_usdt,
                    'target_price': current_price * (1 + tp_pct),
                    'stop_loss_price': current_price * (1 - sl_pct),
                    'reason': f'[] RSI={rsi:.1f}Price trend={price_trend}={price_volatility:.1f}%{amount_usdt:.1f} USDT'
                }
            elif (rsi > 60 and price_trend == 'down' and position.get('amount', 0) > 0):
                # 
                decision = {
                    'action': 'sell',
                    'amount_usdt': position.get('amount', 0) * current_price,
                    'target_price': current_price,
                    'stop_loss_price': current_price,
                    'reason': f'[] RSI={rsi:.1f}Price trend={price_trend}={price_volatility:.1f}%'
                }
            else:
                decision = {
                    'action': 'hold',
                    'amount_usdt': 0,
                    'target_price': current_price,
                    'stop_loss_price': current_price,
                    'reason': f'[] RSI={rsi:.1f}Price trend={price_trend}Bollinger Bands={bb_signal}={volume_signal}={price_volatility:.1f}%'
                }
            
            trading_plan.append({
                'symbol': symbol,
                'decision': decision
            })
        
        return {
            'total_available_usdt': total_available_usdt,
            'trading_plan': trading_plan,
            'reason': f'[] {remaining_usdt:.1f} USDT{len([p for p in trading_plan if p["decision"]["action"] != "hold"])}'
        }
        """LLM"""
        try:
            # 
            prompt = self.build_trading_prompt(market_data)
            
            # LLM API
            if self.selected_llm == 'gpt':
                if self.openai_api_key:
                    decision = self.call_openai_gpt(prompt)
                else:
                    # 
                    decision = self.simulate_gpt_decision(market_data)
                    decision['reason'] = f"[] {decision['reason']}"
            elif self.selected_llm == 'gemini':
                if self.gemini_api_key:
                    decision = self.call_google_gemini(prompt)
                else:
                    # 
                    decision = self.simulate_gemini_decision(market_data)
                    # notesreason
                    decision.notes = f"[] {decision.notes}"
            else:
                # FLAT
                decision = LLMTradingOutput(
                    action="FLAT",
                    entry=Entry(type="market", price=None),
                    stop_loss=StopLoss(price=None, method="swing_or_atr", atr_mult=1.2),
                    take_profit=[],
                    time_in_force="GTC",
                    max_hold_minutes=30,
                    checklist=Checklist(
                        trend_align=False,
                        vol_ok=False,
                        liquidity_ok=True,
                        rr_ok=False,
                        risk_limits_ok=True,
                        news_risk=False
                    ),
                    confidence=0,
                    reason_codes=["NO_LLM_SELECTED"],
                    notes="LLM"
                )
            
            # LLMTradingOutput
            # notes
            return decision
        except Exception as e:
            # FLAT
            return LLMTradingOutput(
                action="FLAT",
                entry=Entry(type="market", price=None),
                stop_loss=StopLoss(price=None, method="swing_or_atr", atr_mult=1.2),
                take_profit=[],
                time_in_force="GTC",
                max_hold_minutes=30,
                checklist=Checklist(
                    trend_align=False,
                    vol_ok=False,
                    liquidity_ok=True,
                    rr_ok=False,
                    risk_limits_ok=True,
                    news_risk=False
                ),
                confidence=0,
                reason_codes=["LLM_ERROR"],
                notes=f"LLM: {str(e)}"
            )
    
    def build_trading_prompt(self, market_data) -> str:
        """Build trading prompt"""
        # Handle both dict and EnhancedLLMTradingInput
        if isinstance(market_data, dict):
            symbol = market_data['symbol']
            current_price = market_data['current_price']
            position = market_data['current_position']
            indicators = market_data['technical_indicators']
            available_usdt = market_data.get('available_usdt', 0)
            total_usdt_balance = market_data.get('total_usdt_balance', 0)
            max_position_size = market_data.get('max_position_size', 100)
            kline_data = market_data.get('kline_data', {})
        else:
            # EnhancedLLMTradingInput object
            symbol = market_data.symbol
            current_price = market_data.prices.close_3m
            position = {'amount': 0, 'value_usdt': 0}
            indicators = market_data.technical_indicators
            available_usdt = 100  # Default value
            total_usdt_balance = 100
            max_position_size = 100
            kline_data = {}
        
        # K203K
        latest_candles = kline_data.get('latest_20_candles', [])
        candles_str = "\n".join([
            f"  K{i+1}: {c.get('open', 0):.2f} {c.get('high', 0):.2f} {c.get('low', 0):.2f} {c.get('close', 0):.2f} {c.get('volume', 0):.0f}"
            for i, c in enumerate(latest_candles[-10:])  # 10
        ])
        
        price_volatility = kline_data.get('price_volatility_20candles', 0)
        avg_candle_range = kline_data.get('avg_candle_range_pct', 0)
        price_change_24h = kline_data.get('price_change_24h', 0)
        
        prompt = f"""
AI Cryptocurrency Trading Bot - 3 Minutes Analysis

Symbol: {symbol}
Current Price: {current_price} USDT
24h Change: {price_change_24h:.2f}%

Account Balance:
- Total USDT: {total_usdt_balance:.2f} USDT
- Available: {available_usdt:.2f} USDT
- Max Position: {max_position_size} USDT

Current Position:
- {symbol.split('/')[0]}: {position.get('amount', 0):.6f}
- Value: {position.get('value_usdt', 0):.2f} USDT

103K30:
{candles_str}

:
- 20K: {price_volatility:.2f}%
- K: {avg_candle_range:.2f}%

:
- RSI: {indicators.get('rsi', 0):.2f} ({indicators.get('rsi_signal', 'neutral')})
- Moving averages: {indicators.get('ma_short', 0):.2f}, {indicators.get('ma_long', 0):.2f}
- Price trend: {indicators.get('price_trend', 'neutral')}
- Bollinger Bands: {indicators.get('bb_signal', 'middle')}
- : {indicators.get('volume_signal', 'normal')}

:
1. 3
2. amount_usdt
3. ****: 0.2%
   - >3%4-6%2.5-3.5%
   - 1-3%3-4%2-2.5%
   - <1%2.5-3%1.5-2%
   - 2.5%
4. K==
5. RSI>70RSI<30
6. hold

JSON:
{{
    "action": "buy/sell/hold",
    "amount_usdt": ,
    "target_price": 2.5%,
    "stop_loss_price": 1.5-3%,
    "reason": "1.K 2. 3."
}}
"""
        return prompt
    
    def simulate_gpt_decision(self, market_data: Dict) -> Dict:
        """Simulate GPT decision (for demo)"""
        indicators = market_data['technical_indicators']
        current_price = market_data['current_price']
        position = market_data['current_position']
        kline_data = market_data.get('kline_data', {})
        
        # 
        price_volatility = kline_data.get('price_volatility_20candles', 2.0)
        avg_candle_range = kline_data.get('avg_candle_range_pct', 1.0)
        
        # 
        rsi = indicators.get('rsi', 50)
        price_trend = indicators.get('price_trend', 'neutral')
        bb_signal = indicators.get('bb_signal', 'middle')
        volume_signal = indicators.get('volume_signal', 'normal')
        
        # 
        available_usdt = market_data.get('available_usdt', 0)
        max_trade_amount = min(available_usdt * 0.2, 10.0)
        
        # 2%
        if price_volatility > 3:
            tp_pct = 0.04   # 4%
            sl_pct = 0.025  # 2.5%
        elif price_volatility > 1.5:
            tp_pct = 0.03   # 3%
            sl_pct = 0.02   # 2%
        else:
            tp_pct = 0.025  # 2.5%
            sl_pct = 0.015  # 1.5%
        
        # 
        if rsi < 30 and price_trend == 'up' and bb_signal == 'lower' and volume_signal == 'high' and available_usdt >= 5:
            return {
                'action': 'buy',
                'amount_usdt': max_trade_amount,
                'target_price': current_price * (1 + tp_pct),
                'stop_loss_price': current_price * (1 - sl_pct),
                'reason': f'[] RSIBollinger Bands{price_volatility:.1f}%{tp_pct*100:.1f}%{sl_pct*100:.1f}%'
            }
        elif rsi > 70 and price_trend == 'down' and bb_signal == 'upper' and volume_signal == 'high':
            return {
                'action': 'sell',
                'amount_usdt': position.get('amount', 0) * current_price,
                'target_price': current_price * (1 - tp_pct),
                'stop_loss_price': current_price * (1 + sl_pct),
                'reason': f'[] RSIBollinger Bands{price_volatility:.1f}%'
            }
        elif rsi < 40 and price_trend == 'up' and available_usdt >= 3:
            return {
                'action': 'buy',
                'amount_usdt': max_trade_amount * 0.5,
                'target_price': current_price * (1 + tp_pct * 0.7),
                'stop_loss_price': current_price * (1 - sl_pct * 0.8),
                'reason': f'[] RSI{price_volatility:.1f}%K{avg_candle_range:.2f}%'
            }
        elif rsi > 60 and price_trend == 'down':
            return {
                'action': 'sell',
                'amount_usdt': position.get('amount', 0) * current_price * 0.5,
                'target_price': current_price * (1 - tp_pct * 0.7),
                'stop_loss_price': current_price * (1 + sl_pct * 0.8),
                'reason': f'[] RSI'
            }
        else:
            return {
                'action': 'hold',
                'amount_usdt': 0,
                'target_price': current_price,
                'stop_loss_price': current_price,
                'reason': f'[] RSI={rsi:.1f}={price_trend}={price_volatility:.1f}%'
            }
    
    def simulate_gemini_decision(self, market_data: Dict) -> LLMTradingOutput:
        """Simulate Gemini decision (returns new format LLMTradingOutput)"""
        indicators = market_data['technical_indicators']
        current_price = market_data['current_price']
        position = market_data['current_position']
        kline_data = market_data.get('kline_data', {})
        available_usdt = market_data.get('available_usdt', 0)
        
        # 
        rsi = indicators.get('rsi', 50)
        rsi_21 = indicators.get('rsi_21', 50)
        macd_line = indicators.get('macd_line', 0)
        macd_signal = indicators.get('macd_signal', 0)
        ema_20 = indicators.get('ema_20', current_price)
        ema_50 = indicators.get('ema_50', current_price)
        bb_position = indicators.get('bb_position', 0.5)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        # 
        price_volatility = kline_data.get('price_volatility_20candles', 2.0)
        avg_candle_range = kline_data.get('avg_candle_range_pct', 1.0)
        
        # 
        trend_bullish = ema_20 > ema_50 and current_price > ema_20
        trend_bearish = ema_20 < ema_50 and current_price < ema_20
        macd_bullish = macd_line > macd_signal and macd_line > 0
        macd_bearish = macd_line < macd_signal and macd_line < 0
        
        # 
        if price_volatility > 2.0:
            tp_pct = 0.04   # 4%
            sl_pct = 0.025  # 2.5%
        elif price_volatility > 1.0:
            tp_pct = 0.025  # 2.5%
            sl_pct = 0.015  # 1.5%
        else:
            tp_pct = 0.02   # 2%
            sl_pct = 0.01   # 1%
        
        # 
        buy_signals = []
        if rsi < 40 and rsi_21 < 45:  # RSI
            buy_signals.append("RSI_OVERSOLD")
        if trend_bullish and macd_bullish:  # MACD
            buy_signals.append("TREND_BULLISH")
        if bb_position < 0.3:  # Bollinger Bands
            buy_signals.append("BB_OVERSOLD")
        if volume_ratio > 1.2:  # 
            buy_signals.append("VOLUME_SPIKE")
        
        # 
        weak_buy_signals = []
        if rsi < 45:  # RSI
            weak_buy_signals.append("RSI_LOW")
        if trend_bullish:  # 
            weak_buy_signals.append("TREND_UP")
        if macd_bullish:  # MACD
            weak_buy_signals.append("MACD_BULLISH")
        if bb_position < 0.4:  # Bollinger Bands
            weak_buy_signals.append("BB_LOW")
        if volume_ratio > 1.0:  # 
            weak_buy_signals.append("VOLUME_OK")
        
        # 
        sell_signals = []
        if rsi > 60 and rsi_21 > 55:  # RSI
            sell_signals.append("RSI_OVERBOUGHT")
        if trend_bearish and macd_bearish:  # MACD
            sell_signals.append("TREND_BEARISH")
        if bb_position > 0.7:  # Bollinger Bands
            sell_signals.append("BB_OVERBOUGHT")
        if volume_ratio > 1.5:  # 
            sell_signals.append("VOLUME_SPIKE")
        
        # 
        if len(buy_signals) >= 2 and available_usdt >= 5 and not position.get('has_position', False):
            # BUY
            return LLMTradingOutput(
                action="BUY",
                entry=Entry(type="market", price=None),
                stop_loss=StopLoss(price=current_price * (1 - sl_pct), method="fixed", atr_mult=1.0),
                take_profit=[TakeProfit(price=current_price * (1 + tp_pct), size_pct=100.0, r_mult=tp_pct/sl_pct)],
                time_in_force="GTC",
                max_hold_minutes=120,  # 
                checklist=Checklist(
                    trend_align=True,
                    vol_ok=True,
                    liquidity_ok=True,
                    rr_ok=True,
                    risk_limits_ok=True,
                    news_risk=False
                ),
                confidence=min(85, 60 + len(buy_signals) * 10),
                reason_codes=buy_signals,
                notes=f"[] ({len(buy_signals)}/4): {', '.join(buy_signals)}RSI={rsi:.1f}, ={'' if trend_bullish else ''}, MACD={'' if macd_bullish else ''}, {price_volatility:.1f}%"
            )
        elif len(sell_signals) >= 2 and position.get('has_position', False):
            # SELL
            return LLMTradingOutput(
                action="SELL",
                entry=Entry(type="market", price=None),
                stop_loss=StopLoss(price=current_price * (1 + sl_pct), method="fixed", atr_mult=1.0),
                take_profit=[TakeProfit(price=current_price * (1 - tp_pct), size_pct=100.0, r_mult=tp_pct/sl_pct)],
                time_in_force="GTC",
                max_hold_minutes=60,
                checklist=Checklist(
                    trend_align=False,
                    vol_ok=True,
                    liquidity_ok=True,
                    rr_ok=True,
                    risk_limits_ok=True,
                    news_risk=False
                ),
                confidence=min(80, 50 + len(sell_signals) * 10),
                reason_codes=sell_signals,
                notes=f"[] ({len(sell_signals)}/4): {', '.join(sell_signals)}RSI={rsi:.1f}, ={'' if trend_bullish else ''}, MACD={'' if macd_bullish else ''}"
            )
        elif len(weak_buy_signals) >= 2 and available_usdt >= 5 and not position.get('has_position', False):
            # 
            return LLMTradingOutput(
                action="BUY",
                entry=Entry(type="market", price=None),
                stop_loss=StopLoss(price=current_price * (1 - sl_pct * 0.8), method="fixed", atr_mult=1.0),
                take_profit=[TakeProfit(price=current_price * (1 + tp_pct * 0.8), size_pct=100.0, r_mult=tp_pct/sl_pct)],
                time_in_force="GTC",
                max_hold_minutes=90,
                checklist=Checklist(
                    trend_align=trend_bullish,
                    vol_ok=volume_ratio > 1.0,
                    liquidity_ok=True,
                    rr_ok=True,
                    risk_limits_ok=True,
                    news_risk=False
                ),
                confidence=50,
                reason_codes=weak_buy_signals,
                notes=f"[] : {', '.join(weak_buy_signals)}RSI={rsi:.1f}, {price_volatility:.1f}%"
            )
        else:
            # HOLD
            signal_summary = f"({len(buy_signals)}): {', '.join(buy_signals) if buy_signals else ''}; ({len(weak_buy_signals)}): {', '.join(weak_buy_signals) if weak_buy_signals else ''}; ({len(sell_signals)}): {', '.join(sell_signals) if sell_signals else ''}"
            return LLMTradingOutput(
                action="HOLD",
                entry=Entry(type="market", price=None),
                stop_loss=StopLoss(price=None, method="swing_or_atr", atr_mult=1.2),
                take_profit=[],
                time_in_force="GTC",
                max_hold_minutes=30,
                checklist=Checklist(
                    trend_align=trend_bullish,
                    vol_ok=volume_ratio > 1.0,
                    liquidity_ok=True,
                    rr_ok=False,
                    risk_limits_ok=True,
                    news_risk=False
                ),
                confidence=20,
                reason_codes=["INSUFFICIENT_SIGNALS"],
                notes=f"[] {signal_summary}RSI={rsi:.1f}, EMA20={ema_20:.2f}, EMA50={ema_50:.2f}, MACD={macd_line:.4f}"
            )
    
    def execute_trading_decision(self, symbol: str, decision: Dict) -> Dict:
        """Execute trade"""
        try:
            action = decision['action'].lower()  # 
            amount_usdt = decision.get('amount_usdt', 0)
            target_price = decision.get('target_price', 0)
            stop_loss_price = decision.get('stop_loss_price', 0)
            
            if action == 'buy' and amount_usdt >= self.min_order_amount:
                import time  # time
                
                print(f" : {symbol} {amount_usdt} USDT")
                
                # 
                current_price = self.get_current_price(symbol)
                if current_price == 0:
                    print(f"  {symbol} ")
                    return {
                        'success': False,
                        'message': f' {symbol} '
                    }
                
                # 
                buy_amount = amount_usdt / current_price
                
                print(f" : {current_price} USDT, : {buy_amount:.8f} {symbol.split('/')[0]}")
                
                # 
                attach_algo_orders = []
                
                # 
                need_tp = target_price > 0 and target_price > current_price * 1.001
                need_sl = stop_loss_price > 0 and stop_loss_price < current_price * 0.999
                
                if need_tp or need_sl:
                    # OKX
                    if need_sl:
                        min_allowed_price = current_price * 0.97
                        if stop_loss_price < min_allowed_price:
                            print(f" {stop_loss_price}{min_allowed_price:.2f}")
                            stop_loss_price = min_allowed_price
                    
                    # OKX
                    if need_tp:
                        tp_order = {
                            'instId': symbol.replace('/', '-'),  # 
                            'tdMode': 'cash',  # 
                            'side': 'sell',
                            'ordType': 'conditional',
                            'sz': str(buy_amount),
                            'tpTriggerPx': str(target_price),
                            'tpOrdPx': '-1',  # 
                            'instType': 'SPOT',  # 
                            'ccy': symbol.split('/')[0]  # 
                        }
                        attach_algo_orders.append(tp_order)
                        print(f" : ={target_price} USDT (+{(target_price/current_price-1)*100:.2f}%)")
                    
                    if need_sl:
                        sl_order = {
                            'instId': symbol.replace('/', '-'),  # 
                            'tdMode': 'cash',  # 
                            'side': 'sell',
                            'ordType': 'conditional',
                            'sz': str(buy_amount),
                            'slTriggerPx': str(stop_loss_price),
                            'slOrdPx': '-1',  # 
                            'instType': 'SPOT',  # 
                            'ccy': symbol.split('/')[0]  # 
                        }
                        attach_algo_orders.append(sl_order)
                        print(f" : ={stop_loss_price} USDT (-{(1-stop_loss_price/current_price)*100:.2f}%)")
                else:
                    print(f" {target_price}{stop_loss_price}")
                
                # 
                order_params = {
                    'accountType': 'spot',
                    'instType': 'SPOT',
                    'tdMode': 'cash'  # 
                }
                
                # 
                buy_result = self.exchange.create_market_buy_order(symbol, buy_amount, {
                    'accountType': 'spot',
                    'instType': 'SPOT',
                    'tdMode': 'cash'
                })
                
                print(f" : {buy_result['id']}, : {buy_amount:.8f}")
                
                # 
                if buy_result['id'] and len(attach_algo_orders) > 0:
                    print(f"⏳ ...")
                    time.sleep(5)  # 5
                    
                    # 
                    try:
                        order_status = self.exchange.fetch_order(buy_result['id'], symbol, params={
                            'accountType': 'spot',
                            'instType': 'SPOT'
                        })
                        
                        if order_status['status'] == 'closed':
                            actual_amount = float(order_status['filled'])
                            print(f" : {actual_amount:.8f} {symbol.split('/')[0]}")
                            
                            # 
                            print(f"⏳ 5{symbol.split('/')[0]}...")
                            time.sleep(5)
                            
                            # 
                            try:
                                balance = self.exchange.fetch_balance()
                                actual_balance = balance.get(symbol.split('/')[0], {}).get('free', 0)
                                print(f" {symbol.split('/')[0]}: {actual_balance:.8f}")
                                
                                # 
                                safe_amount = min(actual_amount, actual_balance)
                                if safe_amount < actual_amount:
                                    print(f" : {safe_amount:.8f} ( {actual_amount:.8f})")
                                
                                # 
                                for algo_order in attach_algo_orders:
                                    algo_order['sz'] = str(safe_amount)
                                    
                            except Exception as e:
                                print(f" : {e}")
                                # 
                                for algo_order in attach_algo_orders:
                                    algo_order['sz'] = str(actual_amount)
                        else:
                            print(f" : {order_status['status']}")
                            return {
                                'success': True,
                                'buy_order_id': buy_result['id'],
                                'message': f': {order_status["status"]}'
                            }
                            
                    except Exception as e:
                        print(f" : {e}")
                        return {
                            'success': True,
                            'buy_order_id': buy_result['id'],
                            'message': f': {str(e)}'
                        }
                    print(f" {len(attach_algo_orders)}")
                    
                    for algo_order in attach_algo_orders:
                        try:
                            # clOrdIdOKX
                            if 'clOrdId' in algo_order:
                                del algo_order['clOrdId']
                            
                            # 
                            order_amount = float(algo_order['sz'])
                            try:
                                final_balance = self.exchange.fetch_balance()
                                available_amount = final_balance.get(symbol.split('/')[0], {}).get('free', 0)
                                
                                if order_amount > available_amount:
                                    print(f"  {order_amount}  {available_amount}")
                                    algo_order['sz'] = str(available_amount)
                                    
                            except Exception as e:
                                print(f" : {e}")
                            
                            print(f" : {algo_order}")
                            
                            # OKXAPI
                            algo_result = self.exchange.private_post_trade_order_algo(algo_order)
                            
                            if algo_result and 'data' in algo_result and len(algo_result['data']) > 0:
                                algo_id = algo_result['data'][0]['algoId']
                                
                                # 
                                order_type = 'take_profit' if 'tpTriggerPx' in algo_order else 'stop_loss'
                                trigger_price = float(algo_order.get('tpTriggerPx', algo_order.get('slTriggerPx', current_price)))
                                
                                # pending_orders
                                self.pending_orders[algo_id] = {
                                    'symbol': symbol,
                                    'type': order_type,
                                    'amount': buy_amount,
                                    'price': trigger_price,
                                    'order_id': algo_id,
                                    'algo_order': True
                                }
                                
                                print(f" {order_type}: {algo_id}")
                            else:
                                print(f" : {algo_result}")
                                
                        except Exception as e:
                            print(f" : {str(e)}")
                
                return {
                    'success': True,
                    'buy_order_id': buy_result['id'],
                    'message': f'{buy_amount:.8f} {symbol.split("/")[0]} ({amount_usdt:.2f} USDT)'
                }
            
            elif action == 'sell':
                print(f" : {symbol}")
                
                # 
                positions = self.get_current_positions()
                position = positions.get(symbol, {})
                available_amount = position.get('amount', 0)
                
                if available_amount > 0:
                    # 
                    current_price = self.get_current_price(symbol)
                    if current_price == 0:
                        print(f"  {symbol} ")
                        return {
                            'success': False,
                            'message': f' {symbol} '
                        }
                    
                    print(f" : {current_price} USDT, : {available_amount:.8f} {symbol.split('/')[0]}")
                    
                    # 
                    sell_result = self.exchange.create_market_sell_order(symbol, available_amount, {
                        'accountType': 'spot',
                        'instType': 'SPOT',
                        'tdMode': 'cash'
                    })
                    
                    print(f" : {sell_result['id']}")
                    
                    # AI
                    if sell_result['id']:
                        print(f" AI: {symbol}")
                        # 1
                        import threading
                        def delayed_analysis():
                            import time
                            time.sleep(1)
                            self._trigger_immediate_analysis(symbol)
                        
                        analysis_thread = threading.Thread(target=delayed_analysis)
                        analysis_thread.daemon = True
                        analysis_thread.start()
                    
                    return {
                        'success': True,
                        'sell_order_id': sell_result['id'],
                        'message': f'{available_amount:.6f} {symbol.split("/")[0]} ({available_amount * current_price:.2f} USDT)AI'
                    }
                else:
                    return {
                        'success': False,
                        'message': f'{symbol.split("/")[0]}'
                    }
            
            else:
                return {
                    'success': True,
                    'message': ''
                }
                
        except Exception as e:
            print(f" : {str(e)}")
            return {
                'success': False,
                'message': f': {str(e)}'
            }
    
    def start_trading_plan(self, llm_type: str) -> Dict:
        """Start trading plan"""
        try:
            self.trading_plan_active = True
            self.selected_llm = llm_type
            self.current_positions = self.get_current_positions()
            
            print(f"\n {llm_type.upper()}")
            
            # AI
            print("\n Starting AI analysis...")
            initial_result = self.run_trading_cycle()
            
            # 
            import threading
            monitor_thread = threading.Thread(target=self._monitor_trading_cycle, daemon=True)
            monitor_thread.start()
            
            #  initial_result 
            if initial_result and initial_result.get('success'):
                print(f"  {len(initial_result.get('results', []))} ")
            
            return {
                'success': True,
                'message': f'{llm_type}',
                'trading_plan_active': True,
                'selected_llm': llm_type,
                'symbols': self.symbols,
                'initial_analysis': initial_result  #  success, results, global_decision
            }
        except Exception as e:
            return {
                'success': False,
                'message': f': {str(e)}'
            }
    
    def stop_trading_plan(self) -> Dict:
        """Stop trading plan and close all positions"""
        try:
            self.trading_plan_active = False
            
            # 
            positions = self.get_current_positions()
            results = []
            
            # 
            for symbol, position in positions.items():
                if position['amount'] > 0:
                    try:
                        sell_result = self.exchange.create_market_sell_order(symbol, 
                            position['amount'], {
                                'accountType': 'spot',
                                'instType': 'SPOT'
                            })
                        results.append({
                            'symbol': symbol,
                            'action': 'force_sell',
                            'amount': position['amount'],
                            'order_id': sell_result['id'],
                            'success': True
                        })
                    except Exception as e:
                        results.append({
                            'symbol': symbol,
                            'action': 'force_sell',
                            'amount': position['amount'],
                            'error': str(e),
                            'success': False
                        })
            
            return {
                'success': True,
                'message': '',
                'trading_plan_active': False,
                'close_results': results
            }
        except Exception as e:
            return {
                'success': False,
                'message': f': {str(e)}'
            }
    
    def run_trading_cycle(self) -> Dict:
        """Run one trading cycle (analyze all symbols at once)"""
        if not self.trading_plan_active:
            return {'success': False, 'message': ''}
        
        print("\n" + "="*80)
        print(" GLOBAL AI ANALYSIS")
        print("="*80)
        
        try:
            # 
            all_market_data = {}
            print(f"\n :")
            for i, symbol in enumerate(self.symbols):
                print(f"    {symbol} ...")
                market_data = self.get_market_data_for_llm(symbol)
                if market_data:
                    all_market_data[symbol] = market_data
                    print(f"    {symbol}: ={market_data.prices.last_price}, ={market_data.position_state.has_position}")
                else:
                    print(f"    {symbol}: ")
                
                # API
                if i < len(self.symbols) - 1:  # 
                    time.sleep(1.0)  # 1
            
            if not all_market_data:
                return {'success': False, 'message': ''}
            
            print(f"\n  {len(all_market_data)} ")
            
            # LLM
            global_decision = self.call_llm_for_global_decision(all_market_data)
            
            print(f"\n Global Decision Summary:")
            print(f"   Total Available: {global_decision.get('total_available_usdt', 0):.2f} USDT")
            print(f"   Symbols Analyzed: {len(global_decision.get('trading_plan', []))}")
            print(f"   Reasoning: {global_decision.get('reason', '')}")
            
            results = []
            
            # 
            for plan in global_decision.get('trading_plan', []):
                symbol = plan['symbol']
                decision = plan['decision']
                
                # AI
                print(f"\n {symbol}")
                market_input = all_market_data[symbol]
                print(f"   Current Price: {market_input.prices.last_price} USDT")
                position_status = "Has Position" if market_input.position_state.has_position else "No Position"
                print(f"   Position: {position_status}")
                print(f"   AI Decision: {decision['action'].upper()}")
                print(f"   Reason: {decision['reason']}")
                if decision.get('amount_usdt', 0) > 0:
                    print(f"   Amount: {decision['amount_usdt']} USDT")
                target_price = decision.get('target_price', 0)
                stop_loss_price = decision.get('stop_loss_price', 0)
                # Handle string values like "N/A"
                if isinstance(target_price, (int, float)) and target_price > 0:
                    print(f"   Target Price: {target_price}")
                if isinstance(stop_loss_price, (int, float)) and stop_loss_price > 0:
                    print(f"   Stop Loss: {stop_loss_price}")
                model = decision.get('llm_model', '')
                timestamp = decision.get('timestamp', '')
                if model:
                    print(f"   Model: {model}")
                if timestamp:
                    print(f"   Timestamp: {timestamp}")
                
                # null
                if decision.get('target_price') is None:
                    decision['target_price'] = 0
                if decision.get('stop_loss_price') is None:
                    decision['stop_loss_price'] = 0
                if decision.get('amount_usdt') is None:
                    decision['amount_usdt'] = 0
                
                # Execute trade
                trade_result = self.execute_trading_decision(symbol, decision)
                
                if trade_result['success']:
                    print(f"    : {trade_result['message']}")
                else:
                    print(f"    : {trade_result['message']}")
                
                # decision
                try:
                    #  LLMTradingOutput 
                    from models import Entry, StopLoss, TakeProfit, Checklist
                    mock_decision = type('obj', (object,), {
                        'action': decision.get('action', 'FLAT'),
                        'confidence': 75,  # 
                        'entry': type('obj', (object,), {'price': decision.get('target_price', 0)})() if decision.get('target_price') else None,
                        'stop_loss': type('obj', (object,), {'price': decision.get('stop_loss_price', 0)})() if decision.get('stop_loss_price') else None,
                        'take_profit': [],
                        'reason': decision.get('reason', '')
                    })()
                    self._add_analysis_to_history('scheduled', symbol, mock_decision, trade_result)
                except:
                    pass
                
                results.append({
                    'symbol': symbol,
                    'decision': decision,
                    'trade_result': trade_result
                })
            
            print("\n" + "="*80)
            print(" AI")
            print("="*80)
            
            return {
                'success': True,
                'results': results,
                'global_decision': global_decision,
                'message': f'{len(results)}'
            }
            
        except Exception as e:
            print(f" Analysis failed: {str(e)}")
            return {'success': False, 'message': f'Analysis failed: {str(e)}'}
    
    def _monitor_trading_cycle(self):
        """Monitor trading cycle in background"""
        import time
        
        print("\n Monitor trading cycle in background...")
        print(" :")
        print("   1. AI")
        print("   2. 30LLM")
        print("   3. 30AI")
        
        last_analysis_time = time.time()
        
        while self.trading_plan_active:
            try:
                # 
                has_pending_orders = len(self.pending_orders) > 0
                
                if has_pending_orders:
                    # 10
                    print(f"\n {datetime.now().strftime('%H:%M:%S')} - {len(self.pending_orders)}")
                    self._check_order_fills()
                    
                    # 30LLM
                    current_time = time.time()
                    elapsed_minutes = (current_time - last_analysis_time) / 60
                    if elapsed_minutes >= 30:
                        print(f"\n {datetime.now().strftime('%H:%M:%S')} - LLM...")
                        self._llm_review_pending_orders()
                        last_analysis_time = current_time
                    
                    time.sleep(10)  # 10
                    
                else:
                    # 
                    current_time = time.time()
                    elapsed_minutes = (current_time - last_analysis_time) / 60
                    
                    if elapsed_minutes >= 30:  # 30
                        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - AI")
                        result = self.run_trading_cycle()
                        
                        if result['success']:
                            print(" AI")
                        else:
                            print(f" AIAnalysis failed: {result.get('message', '')}")
                        
                        last_analysis_time = current_time
                    else:
                        # 
                        remaining_minutes = 30 - int(elapsed_minutes)
                        print(f"\n {datetime.now().strftime('%H:%M:%S')} - {remaining_minutes}...")
                        time.sleep(60)  # 
                
            except Exception as e:
                print(f" : {str(e)}")
                time.sleep(60)  # 1
        
        print(" ")
    
    def _llm_review_pending_orders(self):
        """Let LLM comprehensively evaluate all orders and positions, freely adjust strategy"""
        try:
            print(f"\n{'='*80}")
            print(" LLM")
            print(f"{'='*80}")
            
            # 
            all_symbols = set(self.symbols)
            
            # 
            portfolio_data = {}
            
            print(f"\n  {len(all_symbols)} ...")
            
            for symbol in all_symbols:
                try:
                    # 
                    market_data = self.get_market_data_for_llm(symbol)
                    if not market_data:
                        continue
                    
                    # 
                    symbol_orders = []
                    
                    # 
                    all_exchange_orders = self.get_all_open_orders()
                    if symbol in all_exchange_orders:
                        for order in all_exchange_orders[symbol]:
                            symbol_orders.append({
                                'order_id': order.get('order_id'),
                                'type': order.get('type'),
                                'side': order.get('side'),
                                'price': order.get('price', 0),
                                'amount': order.get('amount', 0),
                                'trigger_price': order.get('trigger_price', 0),
                                'status': order.get('status'),
                                'created_at': order.get('timestamp', 'unknown'),
                                'algo_order': order.get('algo_order', False)
                            })
                    
                    # 
                    position_info = {}
                    if symbol in self.current_positions:
                        pos = self.current_positions[symbol]
                        position_info = {
                            'has_position': True,
                            'amount': pos.get('amount', 0),
                            'entry_price': pos.get('entry_price', 0),
                            'current_value': pos.get('amount', 0) * market_data.prices.last_price,
                            'unrealized_pnl': (market_data.prices.last_price - pos.get('entry_price', 0)) * pos.get('amount', 0),
                            'unrealized_pnl_pct': ((market_data.prices.last_price / pos.get('entry_price', 1)) - 1) * 100,
                            'stop_loss': pos.get('stop_loss'),
                            'tp_levels': pos.get('tp_levels', [])
                        }
                    else:
                        position_info = {'has_position': False}
                    
                    portfolio_data[symbol] = {
                        'market_data': market_data,
                        'orders': symbol_orders,
                        'position': position_info
                    }
                    
                except Exception as e:
                    print(f"    {symbol}:  - {str(e)}")
                    continue
            
            # 
            prompt = self._build_comprehensive_portfolio_prompt(portfolio_data)
            
            # LLM
            print(f"\n LLM...")
            llm_decisions = self._call_llm_for_portfolio_review(prompt)
            
            # LLM
            print(f"\n LLM...")
            self._execute_llm_portfolio_decisions(llm_decisions, portfolio_data)
            
            print(f"\n{'='*80}")
            print(" LLM")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f" LLM: {str(e)}")
    
    def _build_comprehensive_portfolio_prompt(self, portfolio_data: Dict) -> str:
        """Build comprehensive portfolio evaluation prompt"""
        
        # Get account balance
        try:
            balance = self.exchange.fetch_balance()
            total_usdt = balance['USDT']['free']
        except:
            total_usdt = 10000  # 
        
        # 
        total_position_value = 0
        total_unrealized_pnl = 0
        total_orders_count = 0
        
        for symbol, data in portfolio_data.items():
            if data['position']['has_position']:
                total_position_value += data['position'].get('current_value', 0)
                total_unrealized_pnl += data['position'].get('unrealized_pnl', 0)
            total_orders_count += len(data['orders'])
        
        prompt = f"""
AI

===  ===
- USDT: {total_usdt:.2f}
- : {total_position_value:.2f} USDT
- : {total_unrealized_pnl:.2f} USDT ({(total_unrealized_pnl/total_position_value*100 if total_position_value > 0 else 0):.2f}%)
- : {total_orders_count}
- : {total_usdt + total_position_value:.2f} USDT

===  ===
"""
        
        for symbol, data in portfolio_data.items():
            market = data['market_data']
            orders = data['orders']
            position = data['position']
            
            prompt += f"""
--- {symbol} ---
: {market.prices.last_price} USDT

:
  • 15: {market.trend_filters.m15_trend}
  • 1: {market.trend_filters.h1_trend}
  • EMA20 > EMA50: {market.trend_filters.ema20_gt_ema50_m15}
  • ATR: {market.volatility.atr_pct:.2f}%
  • : {market.volatility.vol_state}
  • VWAP: {market.structure_signals.vwap_distance_bp:.0f} 
  • : {market.structure_signals.volume_vs20x:.2f}x

:
"""
            if position['has_position']:
                prompt += f"""  • : {position['amount']:.6f}
  • : {position['entry_price']:.2f} USDT
  • : {position['current_value']:.2f} USDT
  • : {position['unrealized_pnl']:.2f} USDT ({position['unrealized_pnl_pct']:.2f}%)
  • : {position.get('stop_loss', 'N/A')}
  • : {position.get('tp_levels', [])}
"""
            else:
                prompt += "  • \n"
            
            prompt += "\n:\n"
            if orders:
                for order in orders:
                    prompt += f"""  • [{order['side'].upper()}] {order['type']}: {order['amount']:.6f} @ {order['price']:.2f} USDT (ID: {order['order_id']})
"""
            else:
                prompt += "  • \n"
            
            prompt += "\n"
        
        prompt += f"""
===  ===


1. ****: 
2. ****: /
3. ****: 
   - 
   - 
   - 
4. ****: 

===  ===
- EMAVWAP
- 
- 
- 

=== JSON ===
{{
    "overall_strategy": "",
    "actions": [
        {{
            "symbol": "BTC/USDT",
            "action_type": "cancel_order",  //  "create_order", "close_position", "adjust_position"
            "order_id": "order_123",  // cancel_order
            "reason": ""
        }},
        {{
            "symbol": "ETH/USDT",
            "action_type": "create_order",
            "side": "buy",  //  "sell"
            "order_type": "limit",  //  "market"
            "price": 3000.0,  // limit
            "amount": 0.5,  // 
            "stop_loss": 2900.0,  // 
            "take_profit": [3100.0, 3200.0],  // 
            "reason": ""
        }},
        {{
            "symbol": "BTC/USDT",
            "action_type": "close_position",
            "close_percent": 50,  //  (1-100)
            "reason": ""
        }}
    ]
}}


"""
        
        return prompt
    
    def _call_llm_for_portfolio_review(self, prompt: str) -> Dict:
        """Call LLM for portfolio evaluation"""
        try:
            if self.selected_llm == 'gpt' and self.openai_api_key:
                openai.api_key = self.openai_api_key
                response = openai.ChatCompletion.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "AI"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=3000
                )
                llm_response = response.choices[0].message.content.strip()
                
            elif self.selected_llm == 'gemini' and self.gemini_api_key:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt
                )
                llm_response = response.text.strip()
                
            elif self.selected_llm == 'deepseek' and self.deepseek_api_key:
                client = openai.OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "AI"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=3000
                )
                llm_response = response.choices[0].message.content.strip()
            else:
                # 
                return {"overall_strategy": "LLM API", "actions": []}
            
            # JSON
            if llm_response.startswith('```json'):
                llm_response = llm_response[7:]
            if llm_response.startswith('```'):
                llm_response = llm_response[3:]
            if llm_response.endswith('```'):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()
            
            # JSON
            decisions = json.loads(llm_response)
            return decisions
            
        except Exception as e:
            print(f" LLM: {str(e)}")
            # 
            return {"overall_strategy": f"LLM: {str(e)}", "actions": []}
    
    def _execute_llm_portfolio_decisions(self, llm_decisions: Dict, portfolio_data: Dict):
        """Execute LLM portfolio decisions"""
        try:
            print(f"\n LLM: {llm_decisions.get('overall_strategy', '')}")
            
            actions = llm_decisions.get('actions', [])
            if not actions:
                print("   ℹ LLM")
                return
            
            print(f"\n  {len(actions)} :\n")
            
            for i, action in enumerate(actions, 1):
                try:
                    action_type = action.get('action_type', '')
                    symbol = action.get('symbol', '')
                    reason = action.get('reason', '')
                    
                    print(f"{i}. [{symbol}] {action_type.upper()}")
                    print(f"   : {reason}")
                    
                    if action_type == 'cancel_order':
                        # 
                        order_id = action.get('order_id')
                        if order_id and order_id in self.pending_orders:
                            self._cancel_single_order(order_id)
                            print(f"    : {order_id}")
                        else:
                            # order_id
                            self._cancel_symbol_orders(symbol)
                            print(f"     {symbol} ")
                    
                    elif action_type == 'create_order':
                        # 
                        side = action.get('side', 'buy')
                        order_type = action.get('order_type', 'limit')
                        price = action.get('price', 0)
                        amount = action.get('amount', 0)
                        stop_loss = action.get('stop_loss')
                        take_profit = action.get('take_profit', [])
                        
                        if side == 'buy':
                            # 
                            if order_type == 'market':
                                order = self.exchange.create_market_buy_order(symbol, amount)
                            else:
                                order = self.exchange.create_limit_buy_order(symbol, amount, price)
                            
                            print(f"    : {amount} @ {price} USDT")
                            
                            # 
                            order_id = f"llm_buy_{symbol.replace('/', '_')}_{int(time.time())}"
                            self.pending_orders[order_id] = {
                                'symbol': symbol,
                                'type': order_type,
                                'side': side,
                                'price': price,
                                'amount': amount,
                                'status': 'pending',
                                'exchange_order_id': order.get('id'),
                                'timestamp': datetime.now().isoformat(),
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                        
                        else:
                            # 
                            if order_type == 'market':
                                order = self.exchange.create_market_sell_order(symbol, amount)
                            else:
                                order = self.exchange.create_limit_sell_order(symbol, amount, price)
                            
                            print(f"    : {amount} @ {price} USDT")
                    
                    elif action_type == 'close_position':
                        # 
                        close_percent = action.get('close_percent', 100)
                        
                        if symbol in self.current_positions:
                            position = self.current_positions[symbol]
                            position_amount = position.get('amount', 0)
                            close_amount = position_amount * (close_percent / 100)
                            
                            if close_amount > 0:
                                # 
                                order = self.exchange.create_market_sell_order(symbol, close_amount)
                                print(f"     {close_percent}%: {close_amount:.6f} ")
                                
                                # 
                                if close_percent >= 100:
                                    del self.current_positions[symbol]
                                else:
                                    self.current_positions[symbol]['amount'] = position_amount - close_amount
                        else:
                            print(f"    {symbol} ")
                    
                    elif action_type == 'adjust_position':
                        # 
                        new_stop_loss = action.get('new_stop_loss')
                        new_take_profit = action.get('new_take_profit')
                        
                        if symbol in self.current_positions:
                            if new_stop_loss:
                                self.current_positions[symbol]['stop_loss'] = new_stop_loss
                                print(f"    : {new_stop_loss} USDT")
                            
                            if new_take_profit:
                                self.current_positions[symbol]['tp_levels'] = new_take_profit
                                print(f"    : {new_take_profit}")
                        else:
                            print(f"    {symbol} ")
                    
                    print()  # 
                    
                except Exception as e:
                    print(f"    : {str(e)}\n")
                    continue
            
            print(" ")
            
        except Exception as e:
            print(f" LLM: {str(e)}")
    
    def _cancel_single_order(self, order_id: str):
        """Cancel single order"""
        try:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                
                # API
                if order.get('exchange_order_id'):
                    self.exchange.cancel_order(order['exchange_order_id'], order['symbol'])
                
                # pending_orders
                del self.pending_orders[order_id]
                
        except Exception as e:
            print(f"  {order_id}: {str(e)}")
    
    def _build_order_review_prompt(self, symbol: str, market_data: 'LLMTradingInput') -> str:
        """Build order evaluation prompt"""
        prompt = f"""
AI

: {symbol}
: {market_data.prices.last_price} USDT

:
- 15: {market_data.trend_filters.m15_trend}
- 1: {market_data.trend_filters.h1_trend}
- EMA20>EMA50: {market_data.trend_filters.ema20_gt_ema50_m15}
- ATR: {market_data.volatility.atr_pct:.2f}%
- : {market_data.volatility.vol_state}

:
- : {market_data.position_state.has_position}

:
- : {'' if market_data.risk_rules.cooldown_active else ''}
- : {'' if market_data.risk_rules.news_risk else ''}



:
1. 
2. 
3. 
4. 

JSON
{{
    "action": "keep"  "cancel",
    "reason": ""
}}
"""
        return prompt
    
    def _call_llm_for_order_review(self, prompt: str) -> Dict:
        """Call LLM for order evaluation"""
        try:
            if self.selected_llm == 'gpt' and self.openai_api_key:
                openai.api_key = self.openai_api_key
                response = openai.ChatCompletion.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": "AI"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                llm_response = response.choices[0].message.content.strip()
                
            elif self.selected_llm == 'gemini' and self.gemini_api_key:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt
                )
                llm_response = response.text.strip()
                
            elif self.selected_llm == 'deepseek' and self.deepseek_api_key:
                client = openai.OpenAI(
                    api_key=self.deepseek_api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "AI"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                llm_response = response.choices[0].message.content.strip()
            else:
                # 
                return {"action": "keep", "reason": "LLM API"}
            
            # JSON
            if llm_response.startswith('```json'):
                llm_response = llm_response[7:]
            if llm_response.startswith('```'):
                llm_response = llm_response[3:]
            if llm_response.endswith('```'):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()
            
            # JSON
            decision = json.loads(llm_response)
            return decision
            
        except Exception as e:
            print(f" LLM: {str(e)}")
            # 
            return {"action": "keep", "reason": f"LLM: {str(e)}"}
    
    def _cancel_symbol_orders(self, symbol: str):
        """Cancel all orders for specified symbol (including regular and algo orders)"""
        try:
            cancelled_count = 0
            
            # 1. 
            try:
                open_orders = self.safe_api_call(self.exchange.fetch_open_orders, symbol, params={'accountType': 'spot'})
                if open_orders:
                    print(f"    {len(open_orders)} ...")
                    for order in open_orders:
                        try:
                            self.exchange.cancel_order(order['id'], symbol)
                            cancelled_count += 1
                            print(f"    : {order['id']}")
                        except Exception as e:
                            print(f"     {order['id']}: {str(e)}")
            except Exception as e:
                print(f"    : {str(e)}")
            
            # 2. 
            try:
                inst_id = symbol.replace('/', '-')
                for order_type in ['conditional', 'trigger', 'oco', 'move_order_stop']:
                    algo_response = self.safe_api_call(
                        self.exchange.private_get_trade_orders_algo_pending,
                        {'instType': 'SPOT', 'ordType': order_type}
                    )
                    
                    if algo_response and 'data' in algo_response:
                        for algo_order in algo_response['data']:
                            if algo_order.get('instId') == inst_id:
                                try:
                                    # 
                                    cancel_response = self.exchange.private_post_trade_cancel_algos([{
                                        'algoId': algo_order['algoId'],
                                        'instId': inst_id
                                    }])
                                    
                                    if cancel_response and cancel_response.get('code') == '0':
                                        cancelled_count += 1
                                        print(f"    : {algo_order['algoId']} ({order_type})")
                                    else:
                                        print(f"    : {algo_order['algoId']}")
                                except Exception as e:
                                    print(f"    : {str(e)}")
            except Exception as e:
                print(f"    : {str(e)}")
            
            # 3.  pending_orders 
            orders_to_remove = [order_id for order_id, order in list(self.pending_orders.items()) 
                              if order['symbol'] == symbol]
            for order_id in orders_to_remove:
                del self.pending_orders[order_id]
                cancelled_count += 1
            
            if cancelled_count > 0:
                print(f"    {symbol}  {cancelled_count} ")
            else:
                print(f"   {symbol} ")
            
        except Exception as e:
            print(f"  {symbol} : {str(e)}")
    
    def _check_order_fills(self):
        """Check if orders are filled (including algo orders)"""
        try:
            # pending_orderskeys
            order_ids_to_check = list(self.pending_orders.keys())
            
            for order_id in order_ids_to_check:
                if order_id not in self.pending_orders:
                    continue  # 
                    
                order_info = self.pending_orders[order_id]
                symbol = order_info['symbol']
                is_algo_order = order_info.get('algo_order', False)
                
                # ID
                exchange_order_id = order_info.get('exchange_order_id')
                
                try:
                    if is_algo_order:
                        # algoId
                        # order_id"llm_"ID
                        if order_id.startswith('llm_'):
                            continue
                        
                        algo_status = self.exchange.private_get_trade_order_algo({
                            'algoId': order_id,
                            'instType': 'SPOT'
                        })
                        
                        if algo_status and 'data' in algo_status and len(algo_status['data']) > 0:
                            status = algo_status['data'][0].get('state', '')
                            
                            # : effective=, canceled=, order_failed=, filled=
                            if status == 'filled':
                                order_type = order_info['type']
                                print(f" : {symbol} {order_type} @ {order_info['price']} USDT")
                                
                                # pending_orders
                                del self.pending_orders[order_id]
                                
                                # AI
                                print(f" {order_type}AI: {symbol}")
                                self._trigger_immediate_analysis(symbol)
                            
                            elif status in ['canceled', 'order_failed']:
                                print(f" {status}: {symbol} {order_info['type']}")
                                del self.pending_orders[order_id]
                    else:
                        # ID
                        if not exchange_order_id or exchange_order_id.startswith('llm_'):
                            # ID
                            continue
                        
                        order_status = self.exchange.fetch_order(exchange_order_id, symbol, params={
                            'accountType': 'spot',
                            'instType': 'SPOT'
                        })
                        
                        if order_status['status'] == 'closed':
                            print(f" : {symbol} {order_status['side']} {order_status['amount']}")
                            
                            # pending_orders
                            del self.pending_orders[order_id]
                            
                            # AI
                            print(f" AI: {symbol}")
                            self._trigger_immediate_analysis(symbol)
                
                except Exception as e:
                    # 
                    error_msg = str(e)
                    if "Parameter ordId error" not in error_msg:
                        print(f" {order_id}: {error_msg}")
                    continue
                    
        except Exception as e:
            print(f" : {str(e)}")
    
    def _add_analysis_to_history(self, analysis_type: str, symbol: str, decision: 'LLMTradingOutput', execution_result: Dict = None):
        """Add analysis record to history"""
        try:
            analysis_record = {
                'timestamp': datetime.now().isoformat(),
                'type': analysis_type,  # 'immediate', 'scheduled', 'portfolio_review'
                'symbol': symbol,
                'action': decision.action,
                'confidence': decision.confidence,
                'entry_price': decision.entry.price if decision.entry else None,
                'stop_loss': decision.stop_loss.price if decision.stop_loss else None,
                'take_profit': [tp.price for tp in decision.take_profit] if decision.take_profit else [],
                'reason': getattr(decision, 'notes', None) or getattr(decision, 'reason', 'N/A'),
                'execution_result': execution_result
            }
            
            # 
            self.analysis_history.insert(0, analysis_record)
            
            # 
            if len(self.analysis_history) > self.max_history_size:
                self.analysis_history = self.analysis_history[:self.max_history_size]
                
        except Exception as e:
            print(f" : {str(e)}")
    
    def _trigger_immediate_analysis(self, symbol: str):
        """Trigger AI analysis immediately after order fill"""
        try:
            print(f"\n  {symbol}...")
            
            # 
            market_data = self.get_market_data_for_llm(symbol)
            if not market_data:
                return
            
            # AI
            decision = self.call_llm_for_decision(market_data)
            
            # 
            print(f" {symbol} :")
            print(f"   AI: {decision.action.upper()}")
            print(f"   : {decision.confidence}%")
            
            # Execute trade
            execution_result = None
            if decision.action == "BUY" and decision.confidence > 70:
                trade_result = self.execute_trading_decision_from_new_format(symbol, decision)
                execution_result = trade_result
                if trade_result['success']:
                    print(f"    : {trade_result['message']}")
                else:
                    print(f"    : {trade_result['message']}")
            else:
                print(f"    {symbol}: {decision.action} (: {decision.confidence}%) - Execute trade")
            
            # 
            self._add_analysis_to_history('immediate', symbol, decision, execution_result)
                
        except Exception as e:
            print(f" Analysis failed: {str(e)}")
    
    def _execute_sell_decision(self, symbol: str, decision: LLMTradingOutput) -> Dict:
        """Execute sell decision"""
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                return {"success": False, "message": "Get current price"}
            
            # 
            print(f"  {symbol} ...")
            self._cancel_symbol_orders(symbol)
            
            # 
            time.sleep(0.5)
            
            # 
            balance = self.safe_api_call(self.exchange.fetch_balance, params={'accountType': 'spot'})
            base_currency = symbol.split('/')[0]
            
            if base_currency not in balance:
                return {"success": False, "message": f"{symbol}:  {base_currency} "}
            
            # free + used
            total_amount = balance[base_currency].get('total', 0)
            free_amount = balance[base_currency].get('free', 0)
            
            if total_amount <= 0:
                return {"success": False, "message": f"{symbol}: "}
            
            # 
            sell_amount = free_amount
            
            # 
            market = self.exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            
            if sell_amount < min_amount:
                return {
                    "success": False, 
                    "message": f"{symbol}:  {sell_amount:.8f}  {min_amount:.8f}: {total_amount:.8f}"
                }
            
            print(f" : {symbol} {sell_amount:.8f} @ {current_price} (: {total_amount:.8f})")
            
            # 
            sell_order = self.exchange.create_market_sell_order(
                symbol,
                sell_amount,
                params={'tdMode': 'cash'}
            )
            
            print(f": {sell_order['id']}")
            
            # 
            entry_price = current_price
            if symbol in self.current_positions:
                entry_price = self.current_positions[symbol].get('entry_price', current_price)
            
            pnl = (current_price - entry_price) * sell_amount
            pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            # 
            if symbol in self.current_positions:
                del self.current_positions[symbol]
            
            # 
            
            return {
                "success": True,
                "message": f": {sell_amount:.6f} {symbol.split('/')[0]} @ {current_price} | : {pnl:.2f} USDT ({pnl_pct:+.2f}%)",
                "order_id": sell_order['id'],
                "pnl": pnl,
                "pnl_pct": pnl_pct
            }
            
        except Exception as e:
            return {"success": False, "message": f": {str(e)}"}
    
    def execute_trading_decision_from_new_format(self, symbol: str, decision: LLMTradingOutput) -> Dict:
        """Execute new format trading decision (supports BUY and SELL)"""
        try:
            # SELL
            if decision.action == "SELL":
                return self._execute_sell_decision(symbol, decision)
            
            # BUY
            if decision.action != "BUY":
                return {"success": True, "message": f"{symbol}: {decision.action} - Execute trade"}
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                return {"success": False, "message": "Get current price"}
            
            # 5%
            balance = self.exchange.fetch_balance(params={'accountType': 'spot'})
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            amount_usdt = min(usdt_balance * 0.05, 100)  # 100 USDT
            
            if amount_usdt < 5:
                return {"success": False, "message": "USDT"}
            
            buy_amount = amount_usdt / current_price
            
            print(f": {symbol} {buy_amount:.6f} @ {current_price}")
            
            # 
            buy_order = self.exchange.create_market_buy_order(
                symbol, 
                buy_amount,
                params={'tdMode': 'cash'}
            )
            
            print(f": {buy_order['id']}")
            
            # 
            self.current_positions[symbol] = {
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'amount': buy_amount,
                'stop_loss': decision.stop_loss.price,
                'take_profit': [tp.price for tp in decision.take_profit],
                'order_id': buy_order['id']
            }
            
            # 
            self._set_take_profit_stop_loss_orders_from_new_format(symbol, decision, buy_amount)
            
            return {
                "success": True, 
                "message": f": {buy_amount:.6f} {symbol.split('/')[0]} @ {current_price}",
                "order_id": buy_order['id']
            }
            
        except Exception as e:
            return {"success": False, "message": f": {str(e)}"}
    
    def _set_take_profit_stop_loss_orders_from_new_format(self, symbol: str, decision: LLMTradingOutput, amount: float):
        """Set new format take profit/stop loss orders (using OKX algo order API)"""
        try:
            inst_id = symbol.replace('/', '-')  # OKX: BTC-USDT
            
            # OKXAPI
            if decision.stop_loss and decision.stop_loss.price:
                try:
                    stop_loss_response = self.exchange.private_post_trade_order_algo({
                        'instId': inst_id,
                        'tdMode': 'cash',
                        'side': 'sell',
                        'ordType': 'conditional',  # 
                        'sz': str(amount),
                        'triggerPx': str(decision.stop_loss.price),  # 
                        'orderPx': '-1'  # -1
                    })
                    
                    if stop_loss_response and stop_loss_response.get('code') == '0':
                        algo_id = stop_loss_response['data'][0]['algoId']
                        print(f" : {algo_id} @ {decision.stop_loss.price} USDT")
                        
                        # pending_orders
                        self.pending_orders[algo_id] = {
                            'symbol': symbol,
                            'type': 'stop_loss',
                            'side': 'sell',
                            'amount': amount,
                            'price': decision.stop_loss.price,
                            'trigger_price': decision.stop_loss.price,
                            'status': 'effective',
                            'timestamp': datetime.now().isoformat(),
                            'algo_order': True
                        }
                    else:
                        error_msg = stop_loss_response.get('msg', '') if stop_loss_response else ''
                        print(f" : {error_msg}")
                except Exception as e:
                    print(f" : {str(e)}")
            
            # OKXAPI
            if decision.take_profit:
                for i, tp in enumerate(decision.take_profit):
                    try:
                        tp_amount = amount * (tp.size_pct / 100)
                        
                        tp_response = self.exchange.private_post_trade_order_algo({
                            'instId': inst_id,
                            'tdMode': 'cash',
                            'side': 'sell',
                            'ordType': 'conditional',  # 
                            'sz': str(tp_amount),
                            'triggerPx': str(tp.price),  # 
                            'orderPx': '-1'  # -1
                        })
                        
                        if tp_response and tp_response.get('code') == '0':
                            algo_id = tp_response['data'][0]['algoId']
                            print(f"  #{i+1}: {algo_id} {tp_amount:.6f} @ {tp.price} USDT")
                            
                            # pending_orders
                            self.pending_orders[algo_id] = {
                                'symbol': symbol,
                                'type': 'take_profit',
                                'side': 'sell',
                                'amount': tp_amount,
                                'price': tp.price,
                                'trigger_price': tp.price,
                                'status': 'effective',
                                'timestamp': datetime.now().isoformat(),
                                'algo_order': True
                            }
                        else:
                            error_msg = tp_response.get('msg', '') if tp_response else ''
                            print(f" #{i+1}: {error_msg}")
                    except Exception as e:
                        print(f" #{i+1}: {str(e)}")
                
        except Exception as e:
            print(f" : {str(e)}")

