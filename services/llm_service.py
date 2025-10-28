import json
import re
import openai
import google.generativeai as genai
from typing import Dict
from datetime import datetime
from models import LLMTradingInput, LLMTradingOutput


class LLMService:
    """Service for calling LLM APIs for trading decisions"""
    
    def __init__(self):
        self.openai_api_key = None
        self.gemini_api_key = None
        self.deepseek_api_key = None
        self.selected_llm = 'gemini'
    
    def set_api_keys(self, openai_key: str = None, gemini_key: str = None, deepseek_key: str = None):
        """Set LLM API keys"""
        if openai_key:
            self.openai_api_key = openai_key
        if gemini_key:
            self.gemini_api_key = gemini_key
            genai.configure(api_key=gemini_key)
        if deepseek_key:
            self.deepseek_api_key = deepseek_key
    
    def call_openai_gpt(self, prompt: str) -> Dict:
        """Call OpenAI GPT API"""
        try:
            if not self.openai_api_key:
                return {'action': 'hold', 'reason': 'OpenAI API Key not set'}
            
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency trading AI. Make trading decisions based on market data. Only return JSON format decision results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content.strip()
            try:
                decision = json.loads(ai_response)
                return decision
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    return decision
                else:
                    return {'action': 'hold', 'reason': f'AI returned invalid format: {ai_response}'}
                    
        except Exception as e:
            return {'action': 'hold', 'reason': f'OpenAI API call failed: {str(e)}'}
    
    def call_google_gemini(self, prompt: str) -> Dict:
        """Call Google Gemini API"""
        try:
            if not self.gemini_api_key:
                print("Gemini API Key not set, using simulated decision")
                return self._simulate_decision()
            
            from google import genai
            
            client = genai.Client(api_key=self.gemini_api_key)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"You are a professional cryptocurrency trading AI. Make trading decisions based on market data. Only return JSON format decision results, no other text or markdown.\n\n{prompt}",
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 2000,
                }
            )
            
            ai_response = response.text.strip()
            
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            ai_response = ai_response.strip()
            
            try:
                decision = json.loads(ai_response)
                print("JSON parsed successfully")
                return decision
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group())
                        print("Regex extracted JSON successfully")
                        return decision
                    except json.JSONDecodeError:
                        print(f"Regex extraction also failed")
                        return self._simulate_decision()
                else:
                    print(f"No JSON format found")
                    return self._simulate_decision()
                    
        except Exception as e:
            print(f"Gemini API call failed: {str(e)}")
            return self._simulate_decision()
    
    def call_deepseek_v3(self, prompt: str) -> Dict:
        """Call DeepSeek V3 API"""
        try:
            if not self.deepseek_api_key:
                print("DeepSeek API Key not set, using simulated decision")
                return self._simulate_decision()
            
            client = openai.OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a professional cryptocurrency trading AI. Make trading decisions based on market data. Only return JSON format decision results, no other text or markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            ai_response = ai_response.strip()
            
            try:
                decision = json.loads(ai_response)
                print("JSON parsed successfully")
                return decision
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        decision = json.loads(json_str)
                        print("Regex extracted JSON successfully")
                        return decision
                    except:
                        print(f"Regex extraction also failed")
                        return self._simulate_decision()
                else:
                    print(f"No JSON format found")
                    return self._simulate_decision()
                
        except Exception as e:
            print(f"DeepSeek API call failed: {str(e)}")
            return self._simulate_decision()
    
    def get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are a "spot scalping execution auditor" for 3-minute crypto trading. You must follow these iron rules:

1. NEVER invent prices, indicators, or facts
2. Only use the numeric fields provided in the input
3. Output ONLY valid JSON, no explanations
4. Gate checks: If ANY condition fails, return FLAT
5. Position management: Move stop to breakeven after TP1, exit after max hold time
6. You can BUY to enter or add positions
7. You can SELL to exit or reduce positions at any time if market conditions deteriorate"""
    
    def get_developer_prompt(self) -> str:
        """Get developer prompt for LLM"""
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
    
    def _simulate_decision(self) -> Dict:
        """Simulate decision (fallback)"""
        return {
            'action': 'hold',
            'reason': 'Simulated decision (API not configured)',
            'timestamp': datetime.now().isoformat()
        }
    
    def call_llm_with_prompt(self, prompt: str) -> Dict:
        """Call LLM with a prompt"""
        if self.selected_llm == 'gpt':
            return self.call_openai_gpt(prompt)
        elif self.selected_llm == 'gemini':
            return self.call_google_gemini(prompt)
        elif self.selected_llm == 'deepseek':
            return self.call_deepseek_v3(prompt)
        else:
            return self._simulate_decision()

