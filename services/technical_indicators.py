import numpy as np
import pandas as pd
from typing import Dict, Tuple
from models import TechnicalIndicators


class TechnicalIndicatorCalculator:
    """Calculate technical indicators for trading analysis"""
    
    def __init__(self, rsi_period=14, ma_short=5, ma_long=20, bb_period=20, bb_std=2):
        self.rsi_period = rsi_period
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.bb_period = bb_period
        self.bb_std = bb_std
    
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
        """Calculate On-Balance Volume"""
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
        """Calculate True Range"""
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
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
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
        """Calculate moving averages"""
        ma_short = prices.rolling(window=self.ma_short).mean()
        ma_long = prices.rolling(window=self.ma_long).mean()
        return ma_short, ma_long
    
    def calculate_bollinger_bands_series(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands (returns series)"""
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
        """Analyze technical indicators and return signals"""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        volume = df['volume']
        
        rsi = self.calculate_rsi(close, self.rsi_period)
        ma_short, ma_long = self.calculate_moving_averages(close)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        volume_ratio = self.calculate_volume_ratio(volume)
        
        latest_rsi = rsi.iloc[-1]
        latest_ma_short = ma_short.iloc[-1]
        latest_ma_long = ma_long.iloc[-1]
        latest_price = close.iloc[-1]
        latest_bb_upper = bb_upper.iloc[-1]
        latest_bb_lower = bb_lower.iloc[-1]
        latest_volume_ratio = volume_ratio.iloc[-1]
        
        price_trend = "up" if latest_ma_short > latest_ma_long else "down"
        rsi_signal = "oversold" if latest_rsi < 30 else "overbought" if latest_rsi > 70 else "neutral"
        bb_signal = "upper" if latest_price > latest_bb_upper else "lower" if latest_price < latest_bb_lower else "middle"
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
    
    def calculate_enhanced_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate enhanced technical indicators"""
        try:
            rsi_14 = self.calculate_rsi(df['close'], 14).iloc[-1]
            rsi_21 = self.calculate_rsi(df['close'], 21).iloc[-1]
            
            macd_line, macd_signal, macd_histogram = self.calculate_macd(df['close'])
            
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
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else None
            bb_position = (df['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else None
            
            volume_sma_20 = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / volume_sma_20 if volume_sma_20 > 0 else None
            obv = self.calculate_obv(df)
            
            atr_14 = self.calculate_atr(df, 14)
            atr_21 = self.calculate_atr(df, 21)
            
            stochastic_k, stochastic_d = self.calculate_stochastic(df)
            williams_r = self.calculate_williams_r(df)
            cci = self.calculate_cci(df)
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

