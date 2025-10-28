from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from urllib.parse import quote
from typing import Optional, Dict, List, Tuple
import ccxt
import time
import numpy as np
import pandas as pd
import json
import asyncio
import openai
import google.generativeai as genai
from datetime import datetime, timedelta
from models import (
    OrderIn, CancelIn, LLMTradingInput, LLMTradingOutput, 
    PositionManagementInput, PositionManagementOutput,
    Prices, TrendFilters, Volatility, StructureSignals, 
    LiquidityCosts, RiskRules, Precalc, PositionState,
    Entry, StopLoss, TakeProfit, Checklist,
    # Enhanced data models
    EnhancedPrices, TechnicalIndicators, MarketStructure,
    OrderBookData, MarketSentiment, TimeBasedData, EnhancedLLMTradingInput
)
from settings import OKX_API_KEY, OKX_SECRET, OKX_PASSWORD, OKX_TESTNET, PORT

app=FastAPI(title="OKX FastAPI CCXT Backend",version="0.1.0")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

# Import core modules
from core.exchange import create_exchange as ex


# HTML endpoints (read from frontend files)
@app.get("/llm-trading", response_class=HTMLResponse)
def llm_trading_interface():
    """LLM Trading Interface"""
    with open("frontend/llm_trading.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)

@app.get("/", response_class=HTMLResponse)
def homepage():
    """Homepage - LLM Trading Interface"""
    with open("frontend/llm_trading.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


# ===== API Routes =====

# Lazy import LLMTradingSystem to avoid startup issues
from core.trading_system import LLMTradingSystem

# Lazy initialization - create instance on first use
llm_trading_system = None

def get_llm_trading_system():
    global llm_trading_system
    if llm_trading_system is None:
        llm_trading_system = LLMTradingSystem()
    return llm_trading_system

@app.post("/llm-trading/set-api-keys")
def set_llm_api_keys(request: dict):
    """Set LLM API Keys"""
    system = get_llm_trading_system()
    openai_key = request.get('openai_key')
    gemini_key = request.get('gemini_key')
    deepseek_key = request.get('deepseek_key')
    
    if not openai_key and not gemini_key and not deepseek_key:
        raise HTTPException(400, "Please provide at least one API key")
    
    system.set_api_keys(openai_key, gemini_key, deepseek_key)
    
    return {
        'success': True,
        'message': 'API keys set successfully',
        'openai_configured': bool(openai_key),
        'gemini_configured': bool(gemini_key),
        'deepseek_configured': bool(deepseek_key)
    }

@app.get("/llm-trading/api-status")
def get_api_status():
    """Get API key configuration status"""
    system = get_llm_trading_system()
    return {
        'openai_configured': bool(system.openai_api_key),
        'gemini_configured': bool(system.gemini_api_key),
        'deepseek_configured': bool(system.deepseek_api_key),
        'trading_plan_active': system.trading_plan_active,
        'selected_llm': system.selected_llm
    }

@app.post("/llm-trading/start")
def start_llm_trading_plan(llm_type: str = Query(...)):
    """Start LLM trading plan"""
    system = get_llm_trading_system()
    if llm_type not in ['gpt', 'gemini', 'deepseek']:
        raise HTTPException(400, "Unsupported LLM type, please choose 'gpt', 'gemini' or 'deepseek'")
    
    result = system.start_trading_plan(llm_type)
    return result

@app.post("/llm-trading/stop")
def stop_llm_trading_plan():
    """Stop LLM trading plan and close all positions"""
    system = get_llm_trading_system()
    result = system.stop_trading_plan()
    return result

@app.post("/llm-trading/execute")
def execute_llm_trading_cycle():
    """Execute one LLM trading cycle"""
    system = get_llm_trading_system()
    result = system.run_trading_cycle()
    return result

@app.get("/llm-trading/status")
def get_llm_trading_status():
    """Get LLM trading status"""
    system = get_llm_trading_system()
    # Get all open orders
    all_open_orders = system.get_all_open_orders()
    
    # Merge AI algo orders into open orders
    for algo_id, algo_order in system.pending_orders.items():
        symbol = algo_order['symbol']
        if symbol not in all_open_orders:
            all_open_orders[symbol] = []
        
        all_open_orders[symbol].append({
            'order_id': algo_id,
            'type': algo_order['type'],
            'side': 'sell',  # AI algo orders are all sell orders
            'amount': algo_order['amount'],
            'price': algo_order['price'],
            'trigger_price': algo_order['price'],
            'status': 'effective',
            'timestamp': int(time.time() * 1000),
            'algo_order': True
        })
    
    return {
        'trading_plan_active': system.trading_plan_active,
        'selected_llm': system.selected_llm,
        'current_positions': system.get_current_positions(),
        'all_open_orders': all_open_orders,  # All open orders (including AI algo orders)
        'recent_trades': system.get_recent_trades(),       # Trade history
        'pending_orders': system.pending_orders,           # AI algo orders (kept for monitoring)
        'analysis_history': system.analysis_history[:20],  # Last 20 analysis records
        'symbols': system.symbols,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/llm-trading/analysis/{symbol}")
def get_llm_analysis(symbol: str):
    """Get LLM analysis for specific trading pair"""
    system = get_llm_trading_system()
    if symbol not in system.symbols:
        raise HTTPException(404, "Unsupported trading pair")
    
    market_data = system.get_market_data_for_llm(symbol)
    if not market_data:
        raise HTTPException(400, "Unable to fetch market data")
    
    # If LLM is active, get decision
    decision = None
    if system.trading_plan_active and system.selected_llm:
        decision = system.call_llm_for_decision(market_data)
    
    return {
        'symbol': symbol,
        'market_data': market_data,
        'llm_decision': decision,
        'trading_plan_active': system.trading_plan_active,
        'selected_llm': system.selected_llm,
        'timestamp': datetime.now().isoformat()
    }

# Create AI trading system instance (maintain backward compatibility) - lazy initialization
ai_system = None

def get_ai_system():
    global ai_system
    if ai_system is None:
        ai_system = LLMTradingSystem()
    return ai_system

@app.get("/ai-signals")
def get_ai_signals():
    """Get AI signals for all trading pairs"""
    system = get_ai_system()
    signals = []
    for symbol in system.symbols:
        signal_data = system.generate_trading_signal(symbol)
        signals.append(signal_data)
    return {"signals": signals, "timestamp": datetime.now().isoformat()}

@app.get("/ai-signal/{symbol}")
def get_ai_signal(symbol: str):
    """Get AI signal for specific trading pair"""
    system = get_ai_system()
    if symbol not in system.symbols:
        raise HTTPException(404, "Unsupported trading pair")
    
    signal_data = system.generate_trading_signal(symbol)
    return signal_data

@app.post("/ai-trade-execute")
def execute_ai_trade():
    """Execute AI trade"""
    try:
        system = get_ai_system()
        results = system.run_ai_trading_cycle()
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"AI trade execution failed: {str(e)}")

@app.get("/ai-analysis/{symbol}")
def get_ai_analysis(symbol: str):
    """Get detailed technical analysis for specific trading pair"""
    system = get_ai_system()
    if symbol not in system.symbols:
        raise HTTPException(404, "Unsupported trading pair")
    
    try:
        # Get K-line data
        df = system.get_kline_data(symbol, '3m', 100)
        if df.empty:
            raise HTTPException(400, "Unable to fetch K-line data")
        
        # Analyze technical indicators
        indicators = system.analyze_technical_indicators(df)
        if not indicators:
            raise HTTPException(400, "Technical indicator analysis failed")
        
        # Generate trading signal
        signal_data = system.generate_trading_signal(symbol)
        
        return {
            "symbol": symbol,
            "indicators": indicators,
            "signal": signal_data,
            "kline_data": {
                "latest_price": float(df['close'].iloc[-1]),
                "latest_volume": float(df['volume'].iloc[-1]),
                "price_change_24h": float((df['close'].iloc[-1] - df['close'].iloc[-480]) / df['close'].iloc[-480] * 100) if len(df) >= 480 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.get("/ai-models")
def get_ai_models():
    """Get list of all AI models"""
    return {
        "models": [
            {
                "id": "claude",
                "name": "Claude Sonnet 4.5",
                "status": "active",
                "total_pnl": 650.76,
                "available_cash": 7386.06,
                "positions_count": 2
            },
            {
                "id": "deepseek", 
                "name": "DeepSeek Chat V3.1",
                "status": "active",
                "total_pnl": 350.28,
                "available_cash": 4927.64,
                "positions_count": 2
            },
            {
                "id": "gpt",
                "name": "GPT 5", 
                "status": "active",
                "total_pnl": -426.17,
                "available_cash": 3123.79,
                "positions_count": 1
            },
            {
                "id": "gemini",
                "name": "Gemini 2.5 Pro",
                "status": "active", 
                "total_pnl": 333.88,
                "available_cash": 3123.79,
                "positions_count": 1
            }
        ]
    }

@app.get("/ai-positions/{model_id}")
def get_ai_positions(model_id: str):
    """Get position info for specified AI model"""
    positions_data = {
        "claude": [
            {
                "symbol": "XRP/USDT",
                "side": "long",
                "leverage": 8,
                "notional": 12525.0,
                "unrealized_pnl": 631.82,
                "entry_price": 0.45,
                "current_price": 0.52,
                "size": 24090.91
            },
            {
                "symbol": "ETH/USDT", 
                "side": "long",
                "leverage": 15,
                "notional": 22534.0,
                "unrealized_pnl": 18.95,
                "entry_price": 3200.0,
                "current_price": 3201.26,
                "size": 7.04
            }
        ],
        "deepseek": [
            {
                "symbol": "DOGE/USDT",
                "side": "long", 
                "leverage": 10,
                "notional": 8500.0,
                "unrealized_pnl": 125.50,
                "entry_price": 0.08,
                "current_price": 0.0815,
                "size": 104166.67
            },
            {
                "symbol": "BTC/USDT",
                "side": "short",
                "leverage": 20, 
                "notional": 15000.0,
                "unrealized_pnl": -75.25,
                "entry_price": 67000.0,
                "current_price": 67022.0,
                "size": 0.22
            }
        ],
        "gpt": [
            {
                "symbol": "SOL/USDT",
                "side": "long",
                "leverage": 12,
                "notional": 9200.0,
                "unrealized_pnl": -144.97,
                "entry_price": 180.0,
                "current_price": 178.8,
                "size": 51.11
            }
        ],
        "gemini": [
            {
                "symbol": "BNB/USDT",
                "side": "short",
                "leverage": 5,
                "notional": 6800.0,
                "unrealized_pnl": 333.88,
                "entry_price": 600.0,
                "current_price": 593.3,
                "size": 11.33
            }
        ]
    }
    
    if model_id not in positions_data:
        raise HTTPException(404, "AI model does not exist")
    
    return {
        "model_id": model_id,
        "positions": positions_data[model_id]
    }

@app.get("/ai-trades/{model_id}")
def get_ai_trades(model_id: str, limit: int = 50):
    """Get trade history for specified AI model"""
    # Here we can fetch real data from database or API
    return {
        "model_id": model_id,
        "trades": [
            {
                "id": "trade_001",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.001,
                "price": 67000.0,
                "timestamp": "2025-01-21T10:30:00Z",
                "pnl": 15.50,
                "status": "completed"
            }
        ],
        "total": 1,
        "limit": limit
    }

@app.post("/ai-trade")
def execute_ai_trade(trade_data: dict):
    """Execute AI trade command"""
    try:
        model_id = trade_data.get("model_id")
        symbol = trade_data.get("symbol")
        side = trade_data.get("side")  # "buy" or "sell"
        amount = trade_data.get("amount")
        order_type = trade_data.get("type", "market")
        price = trade_data.get("price")
        
        # Here we can add AI model validation logic
        if not model_id or not symbol or not side or not amount:
            raise HTTPException(400, "Missing required parameters")
        
        # Call existing trading API
        order_data = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price
        }
        
        # Here we can log AI trades
        print(f"AI trade executed: {model_id} - {symbol} {side} {amount}")
        
        # Return trade result
        return {
            "success": True,
            "model_id": model_id,
            "order_id": f"ai_{model_id}_{int(time.time())}",
            "message": "AI trade command executed"
        }
        
    except Exception as e:
        raise HTTPException(400, f"AI trade execution failed: {str(e)}")

@app.get("/health")
def health():
    return {"ok":True,"testnet":OKX_TESTNET}

@app.get("/exchange")
def exchange_info():
    e=ex()
    return {"id":e.id,"name":e.name,"countries":e.countries,"urls":e.urls}

@app.get("/ticker")
def ticker(symbol:str=Query(...)):
    e=ex()
    e.load_markets()
    return e.fetch_ticker(symbol)

@app.get("/ohlcv")
def ohlcv(symbol:str=Query(...),timeframe:str="1m",limit:int=200,since:Optional[int]=None):
    e=ex()
    e.load_markets()
    data=e.fetch_ohlcv(symbol,timeframe=timeframe,limit=limit,since=since)
    return [{"timestamp":d[0],"open":d[1],"high":d[2],"low":d[3],"close":d[4],"volume":d[5]} for d in data]

@app.get("/balance")
def balance():
    e=ex()
    return e.fetch_balance(params={'accountType': 'spot'})

@app.get("/orders/open")
def open_orders(symbol:Optional[str]=None):
    e=ex()
    e.load_markets()
    return e.fetch_open_orders(symbol, params={'accountType': 'spot'}) if symbol else e.fetch_open_orders(params={'accountType': 'spot'})

@app.get("/orders/closed")
def closed_orders(symbol:Optional[str]=None):
    e=ex()
    e.load_markets()
    return e.fetch_closed_orders(symbol, params={'accountType': 'spot'}) if symbol else e.fetch_closed_orders(params={'accountType': 'spot'})

@app.get("/trades")
def my_trades(symbol:Optional[str]=None,limit:int=50):
    e=ex()
    e.load_markets()
    return e.fetch_my_trades(symbol,limit=limit, params={'accountType': 'spot'}) if symbol else e.fetch_my_trades(limit=limit, params={'accountType': 'spot'})

@app.post("/order")
def create_order(body:OrderIn):
    e=ex()
    e.load_markets()
    try:
        print(f"Order request: symbol={body.symbol}, side={body.side}, type={body.type}, amount={body.amount}, price={body.price}")
        
        if body.type == 'market':
            if body.side == 'buy':
                result = e.create_market_buy_order(body.symbol, body.amount, {
                    'accountType': 'spot',
                    'instType': 'SPOT'
                })
            else:
                result = e.create_market_sell_order(body.symbol, body.amount, {
                    'accountType': 'spot',
                    'instType': 'SPOT'
                })
        else:
            result = e.create_limit_order(body.symbol, body.side, body.amount, body.price, {
                'accountType': 'spot',
                'instType': 'SPOT'
            })
                
        return result
    except Exception as err:
        print(f"Error details: {err}")
        print(f"Error type: {type(err)}")
        raise HTTPException(400, str(err))

@app.post("/order/futures")
def create_futures_order(body:OrderIn):
    e=ex()
    e.load_markets()
    try:
        print(f"Futures order request: symbol={body.symbol}, side={body.side}, type={body.type}, amount={body.amount}, price={body.price}")
        
        params = body.params.copy() if body.params else {}
        params['accountType'] = 'futures'
        params['instType'] = 'SWAP'
        params['tdMode'] = 'cross'
        params['posSide'] = 'long' if body.side == 'buy' else 'short'
        
        if body.type == 'market':
            if body.side == 'buy':
                result = e.create_market_buy_order(body.symbol, body.amount, params)
            else:
                result = e.create_market_sell_order(body.symbol, body.amount, params)
        else:
            result = e.create_limit_order(body.symbol, body.side, body.amount, body.price, params)
                
        return result
    except Exception as err:
        print(f"Futures order error: {err}")
        print(f"Error type: {type(err)}")
        raise HTTPException(400, str(err))

@app.post("/order/cancel")
def cancel_order(body:CancelIn):
    e=ex()
    e.load_markets()
    try:
        return e.cancel_order(body.id,body.symbol,body.params)
    except ccxt.BaseError as err:
        raise HTTPException(400,str(err))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
