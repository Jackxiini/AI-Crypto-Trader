from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class OrderIn(BaseModel):
    symbol: str
    side: str
    type: str = Field(default="market")
    amount: float
    price: Optional[float] = None
    params: dict = Field(default_factory=dict)

class CancelIn(BaseModel):
    id: str
    symbol: Optional[str] = None
    params: dict = Field(default_factory=dict)

# Enhanced LLM data models
class EnhancedPrices(BaseModel):
    last_price: float
    candidate_entry: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    open_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_percent_24h: Optional[float] = None

class TechnicalIndicators(BaseModel):
    # Trend indicators
    rsi_14: Optional[float] = None
    rsi_21: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Moving averages
    sma_5: Optional[float] = None
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_5: Optional[float] = None
    ema_10: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_100: Optional[float] = None
    ema_200: Optional[float] = None
    
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_position: Optional[float] = None  # 0-1 scale, 0.5 is midline
    
    # Volume indicators
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None  # Current volume / 20-day average
    obv: Optional[float] = None  # On-Balance Volume
    
    # Volatility indicators
    atr_14: Optional[float] = None
    atr_21: Optional[float] = None
    bollinger_width: Optional[float] = None
    
    # Momentum indicators
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None  # Commodity Channel Index
    
    # Trend strength
    adx: Optional[float] = None  # Average Directional Index
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None

class MarketStructure(BaseModel):
    # Support and resistance levels
    support_levels: List[float] = []
    resistance_levels: List[float] = []
    pivot_point: Optional[float] = None
    pivot_resistance_1: Optional[float] = None
    pivot_resistance_2: Optional[float] = None
    pivot_support_1: Optional[float] = None
    pivot_support_2: Optional[float] = None
    
    # Price structure
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False
    trend_strength: str = "sideways"  # Options: strong_up, up, sideways, down, strong_down
    
    # Key price levels
    recent_swing_low: Optional[float] = None
    recent_swing_high: Optional[float] = None
    swing_low_time: Optional[str] = None
    swing_high_time: Optional[str] = None

class OrderBookData(BaseModel):
    # Order book depth
    bid_depth_1: Optional[float] = None
    bid_depth_5: Optional[float] = None
    bid_depth_10: Optional[float] = None
    ask_depth_1: Optional[float] = None
    ask_depth_5: Optional[float] = None
    ask_depth_10: Optional[float] = None
    
    # Buy/sell pressure
    bid_ask_ratio: Optional[float] = None
    order_book_imbalance: Optional[float] = None  # Positive = buy pressure, negative = sell pressure
    
    # Large order information
    large_bid_orders: int = 0
    large_ask_orders: int = 0
    large_order_threshold: float = 10000.0  # USDT

class MarketSentiment(BaseModel):
    # Market sentiment indicators
    fear_greed_index: Optional[float] = None  # Range 0-100
    social_sentiment: Optional[str] = None  # Options: bullish, bearish, neutral
    news_sentiment: Optional[str] = None  # Options: positive, negative, neutral
    
    # Money flow
    money_flow_index: Optional[float] = None
    accumulation_distribution: Optional[float] = None
    
    # Market participation
    active_addresses: Optional[int] = None
    transaction_count: Optional[int] = None
    network_value: Optional[float] = None

class TimeBasedData(BaseModel):
    # Time period analysis
    current_time: str
    market_session: str  # Options: asian, european, american, overlap
    volatility_by_hour: Dict[str, float] = {}
    volume_by_hour: Dict[str, float] = {}
    
    # Historical patterns
    same_time_yesterday: Optional[float] = None
    same_time_last_week: Optional[float] = None
    same_time_last_month: Optional[float] = None
    
    # Cyclical analysis
    day_of_week: int  # Range 0-6
    hour_of_day: int  # Range 0-23
    is_weekend: bool = False
    is_holiday: bool = False

class EnhancedLLMTradingInput(BaseModel):
    # Basic information
    symbol: str
    timeframe: str
    now_utc: str
    
    # Enhanced data
    prices: EnhancedPrices
    technical_indicators: TechnicalIndicators
    market_structure: MarketStructure
    order_book: OrderBookData
    market_sentiment: MarketSentiment
    time_data: TimeBasedData
    
    # Legacy data (maintained for compatibility)
    trend_filters: 'TrendFilters'
    volatility: 'Volatility'
    structure_signals: 'StructureSignals'
    liquidity_costs: 'LiquidityCosts'
    risk_rules: 'RiskRules'
    precalc: 'Precalc'
    position_state: 'PositionState'

class Prices(BaseModel):
    last_price: float
    candidate_entry: Optional[float] = None

class TrendFilters(BaseModel):
    m15_trend: str  # up/down/sideways
    h1_trend: str
    ema20_gt_ema50_m15: bool
    ema20_slope_3m: float

class Volatility(BaseModel):
    atr_3m_14: float
    atr_pct: float
    vol_state: str  # low/ok/high
    min_atr_pct: float
    max_atr_pct: float

class StructureSignals(BaseModel):
    recent_swing_low: float
    recent_swing_high: float
    broke_recent_high_n: bool
    vwap_distance_bp: float
    volume_vs20x: float

class LiquidityCosts(BaseModel):
    spread_bp: float
    est_slippage_bp: float
    taker_fee_bp: float
    maker_fee_bp: float
    costs_ok: bool
    cost_share_of_edge: float

class RiskRules(BaseModel):
    account_equity: float
    max_risk_pct: float
    min_rr: float
    day_max_drawdown_hit: bool
    cooldown_active: bool
    news_risk: bool
    atr_mult: float
    breakeven_after_tp1: bool
    tp_policy: List[float]
    tp_size_pct: List[float]
    max_hold_minutes: int

class Precalc(BaseModel):
    swing_stop: float
    atr_stop: float
    chosen_stop: float
    r_per_dollar: Optional[float] = None
    tp_prices: Optional[List[float]] = None
    rr_ok: bool
    trend_align: bool
    vol_ok: bool
    liquidity_ok: bool
    risk_limits_ok: bool

class PositionState(BaseModel):
    has_position: bool
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    tp_levels: Optional[List[float]] = None
    tp_filled: Optional[List[bool]] = None
    remaining_size_pct: Optional[float] = None
    minutes_in_position: Optional[int] = None
    # Additional fields
    current_amount: Optional[float] = None
    current_value_usdt: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    has_stop_loss: Optional[bool] = None
    has_take_profit: Optional[bool] = None
    open_orders_count: Optional[int] = None

class LLMTradingInput(BaseModel):
    symbol: str
    timeframe: str
    now_utc: str
    prices: 'Prices'
    trend_filters: TrendFilters
    volatility: Volatility
    structure_signals: StructureSignals
    liquidity_costs: LiquidityCosts
    risk_rules: RiskRules
    precalc: Precalc
    position_state: PositionState

# LLM output protocol
class Entry(BaseModel):
    type: str  # market/limit
    price: Optional[float] = None

class StopLoss(BaseModel):
    price: Optional[float] = None
    method: str
    atr_mult: float

class TakeProfit(BaseModel):
    price: float
    size_pct: float
    r_mult: float

class Checklist(BaseModel):
    trend_align: bool
    vol_ok: bool
    liquidity_ok: bool
    rr_ok: bool
    risk_limits_ok: bool
    news_risk: bool

class LLMTradingOutput(BaseModel):
    action: str  # Options: BUY/SELL/FLAT/HOLD/EXIT
    entry: Entry
    stop_loss: StopLoss
    take_profit: List[TakeProfit]
    time_in_force: str  # Options: GTC/IOC
    max_hold_minutes: int
    checklist: Checklist
    confidence: int
    reason_codes: List[str]
    notes: str

# Position management input
class PositionManagementInput(BaseModel):
    symbol: str
    entry_price: float
    current_price: float
    stop_loss: float
    tp_levels: List[float]
    tp_filled: List[bool]
    remaining_size_pct: float
    atr_3m_14: float
    recent_swing_low: float
    minutes_in_position: int
    policy: Dict[str, Any]

# Position management output
class PositionManagementOutput(BaseModel):
    action: str
    move_stop: Optional[Dict[str, float]] = None
    take_more: List[Dict[str, Any]] = []
    close_all: bool = False
    reason_codes: List[str]
    notes: str
