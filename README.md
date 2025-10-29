<h1 align="center">AI Crypto Trading Platform</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/github/Jackxiini/Crypto-AI-Trading/LICENSE" alt="license">
</p>

A cryptocurrency trading automation system powered by Large Language Models (LLM) for OKX exchange.

Now, it is only for OKX exchange but I am still working on it so in the future this will support more exchanges and I will make more features.

## Supported Trading Pairs

- BTC/USDT
- ETH/USDT
- BNB/USDT
- XRP/USDT
- SOL/USDT
- DOGE/USDT
- more in the future

## Features

- **AI-Powered Trading**: Automated trading decisions using GPT-5, Google Gemini 2.5, or DeepSeek V3.1
- **Real-time Analysis**: Technical indicators (RSI, MACD, Bollinger Bands, OBV, ATR, etc.)
- **Auto Trading**: Configurable automated trading cycles with stop loss/take profit
- **Position Management**: Automatic position tracking and risk management

## Project Structure

```
okx_fastapi_backend/
├── main.py                     # Main FastAPI application
├── models.py                   # Pydantic data models
├── settings.py                 # Environment configuration
├── requirements.txt            # Python dependencies
│
├── core/                       # Core functionality
│   ├── exchange.py             # Exchange connection & rate limiting
│   └── trading_system.py       # LLMTradingSystem - Main trading logic
│
├── services/                   # Business logic services
│   ├── technical_indicators.py # Technical indicator calculations
│   └── llm_service.py          # LLM API integration
│
└── frontend/                   # Frontend HTML files
    └── llm_trading.html        # LLM trading interface (Homepage)
```

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/Jackxiini/Crypto-AI-Trading.git

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and edit with your OKX credentials:

```env
OKX_API_KEY=your_api_key
OKX_SECRET=your_secret
OKX_PASSWORD=your_passphrase
OKX_TESTNET=true  # Set to false for live trading
PORT=8000
```

## How to Use

### 1. Start the Server

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at: **http://localhost:8000**

### 2. Access the Web Interface

Open your browser and navigate to: **http://localhost:8000**

### 3. Set Up API Keys

1. In the **API Configuration** section, enter your LLM API keys:
   - **OpenAI API Key** (for GPT-5)
   - **Gemini API Key** (for Google Gemini 2.5 Flash)
   - **DeepSeek API Key** (for DeepSeek V3.1)
2. Click **"Set API Keys"** button
3. Verify the status indicators show "✓ Configured" for your selected model

### 4. Select LLM Model

Choose your preferred AI model by clicking on one of the model cards:
- **GPT-5** - OpenAI
- **Gemini 2.5 Flash** - Google AI 
- **DeepSeek V3.1** - DeepSeek (recommended)

### 5. Start Automated Trading

Click **"Start Trading"** button to begin the automated trading system.

#### Trading Rules and Behavior:

- **Global Analysis**: The AI analyzes all 6 supported trading pairs (BTC, ETH, BNB, XRP, SOL, DOGE) simultaneously
- **Decision Making**: The AI evaluates market conditions, technical indicators, and risk factors to make trading decisions
- **Auto Trading Cycle**: 
  - Initial analysis runs immediately upon start
  - Every **30 minutes**, the AI automatically re-analyzes all markets and adjusts positions
  - The system monitors your positions continuously and makes decisions based on market conditions
- **Risk Management**: 
  - Configurable stop loss and take profit levels
  - Position size limits
  - Volatility-based risk assessment

### 6. Stop Automated Trading

Click **"Stop Trading"** button to:
- Stop all automated trading cycles
- **Automatically close all open positions**
- Cancel all pending orders
- Return control to manual trading

## Trading Workflow

```
1. Start Trading
   ↓
2. AI analyzes all markets (BTC, ETH, BNB, XRP, SOL, DOGE)
   ↓
3. Global decision is made based on:
   - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Market structure and trends
   - Volatility conditions
   - Current positions
   ↓
4. Actions are executed:
   - BUY: Open new positions with stop loss/take profit
   - SELL: Close positions
   - HOLD: Maintain current positions
   ↓
5. Wait 30 minutes
   ↓
6. Repeat analysis and decision making cycle
```

## API Endpoints

### Core Trading

```
GET  /health                  # Health check
GET  /exchange                # Exchange info
GET  /ticker?symbol=BTC/USDT  # Get ticker data
GET  /ohlcv?symbol=BTC/USDT&timeframe=1m
GET  /balance                 # Account balance
GET  /orders/open             # Open orders
POST /order                   # Create order
```

### LLM Trading Automation

```
POST /llm-trading/set-api-keys          # Set LLM API keys
GET  /llm-trading/api-status            # Check API key status
POST /llm-trading/start?llm_type=gemini # Start trading (gemini/gpt/deepseek)
POST /llm-trading/stop                  # Stop trading & close positions
POST /llm-trading/execute               # Execute one manual cycle
GET  /llm-trading/status                # Get trading status & positions
GET  /llm-trading/analysis/{symbol}     # Get detailed analysis
```

### Web Interfaces

```
GET /                  # LLM Trading Interface (Homepage)
GET /llm-trading       # LLM Trading Interface (alternative)
GET /docs              # API Documentation (Swagger UI)
```

## Requirements

- Python 3.10+
- OKX API credentials
- At least one LLM API key (OpenAI, Gemini, or DeepSeek)
- Virtual environment (recommended)

## Important Notes

⚠️ **Risk Warning**: Automated trading involves financial risk. Always test with `OKX_TESTNET=true` before live trading.

⚠️ **API Costs**: LLM API calls have costs. Monitor your usage to avoid unexpected charges.

⚠️ **Market Conditions**: The AI may decide to HOLD positions during unfavorable market conditions (bearish trends, low volatility).


## License
MIT License

## Support

For issues, questions, or contributions, please open an issue on GitHub.
