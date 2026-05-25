# ai_detector.py
import json
import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Resolve and load environment variables from parent directory's .env file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, ".env")
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

EURI_API_KEY = os.getenv("EURI_API_KEY")
EURI_BASE_URL = os.getenv("EURI_BASE_URL")

def _parse_ai_json(content: str) -> dict:
    """
    Cleans up any potential markdown formatting wrapping the JSON string and parses it.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    return json.loads(content)

def detect_chart_pattern(symbol: str, df: pd.DataFrame) -> dict:
    """
    Slices the last 30 trading days of OHLCV data for a stock, formats it, 
    and sends it to the primary Euri AI (gpt-4.1-mini) model.
    Falls back gracefully to the Groq cascade list of models if Euri API fails.
    """
    if df is None or len(df) < 30:
        return {
            "pattern_name": "None",
            "confidence": "None",
            "direction": "None",
            "analysis_text": "Insufficient historical data available. Needs at least 30 trading days of history."
        }
        
    # Take only the last 30 trading days of data
    df_subset = df.iloc[-30:].copy()
    data_lines = []
    
    # Format standard compact data representation
    for _, row in df_subset.iterrows():
        date_str = row['Date'].strftime("%Y-%m-%d")
        data_lines.append(
            f"{date_str} | O: {row['Open']:.2f} | H: {row['High']:.2f} | L: {row['Low']:.2f} | C: {row['Close']:.2f} | V: {int(row['Volume'])}"
        )
    data_str = "\n".join(data_lines)
    
    system_prompt = (
       "You are an expert technical analyst specializing in classical chart pattern recognition. "
    "Analyze the provided 30-day daily OHLCV sequence and identify any classical price patterns present. "
    
    "Patterns to detect include: Double Bottom, Double Top, Head & Shoulders, Inverse Head & Shoulders, "
    "Cup & Handle, Ascending Triangle, Descending Triangle, Symmetrical Triangle, "
    "Bull Flag, Bear Flag, Pennant, Rising Wedge, Falling Wedge, Channel, Rectangle. "
    
    "If no recognizable pattern is present, return pattern as 'None'. "
    "Do not hallucinate patterns — only report what is clearly supported by the price structure. "
    
    "Respond STRICTLY with a single valid JSON object in this exact schema, with no other text:\n"
    "{\n"
    '  "pattern_name": "<pattern name or None>",\n'
    '  "confidence": "<High | Medium | Low | None>",\n'
    '  "direction": "<Bullish | Bearish | Neutral | None>",\n'
    '  "analysis_text": "<2-3 sentence explanation>"\n'
    "}"
    )
    
    user_prompt = f"""
    Analyze the following 30 trading days of daily OHLCV data for {symbol} and identify any classical chart pattern.
    
    Daily OHLCV Data (CSV format — Date, Open, High, Low, Close, Volume):
    {data_str}

    Instructions:
    - Detect one dominant pattern only. If no clear pattern exists, set pattern_name to "None".
    - Do not hallucinate patterns. Only report what is clearly supported by price structure.
    - Return ONLY a raw JSON object — no markdown, no code fences, no explanation outside the JSON.

    Return exactly this JSON structure:
    {{
    "pattern_name": "e.g. Double Bottom | Ascending Triangle | Bullish Flag | None",
    "confidence": "High | Medium | Low | None",
    "direction": "Bullish | Bearish | Neutral | None",
    "analysis_text": "2-3 sentences covering key support/resistance levels, breakout criteria, and target zone if applicable."
    }}
    """
    
    last_error = None
    
    # --- Try Euri API first (gpt-4.1-mini) ---
    if EURI_API_KEY:
        # Resolve EURI Base URL
        if EURI_BASE_URL:
            if EURI_BASE_URL.endswith("/chat/completions"):
                euri_url = EURI_BASE_URL
            else:
                euri_url = EURI_BASE_URL.rstrip("/") + "/chat/completions"
        else:
            euri_url = "https://api.euron.one/api/v1/euri/chat/completions"
            
        euri_headers = {
            "Authorization": f"Bearer {EURI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        euri_payload = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        try:
            print(f"Sending pattern analysis request for {symbol} to Euri API (gpt-4.1-mini)...")
            response = requests.post(euri_url, headers=euri_headers, json=euri_payload, timeout=25)
            if response.status_code == 200:
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content'].strip()
                parsed = _parse_ai_json(content)
                
                return {
                    "pattern_name": parsed.get("pattern_name", "None"),
                    "confidence": parsed.get("confidence", "None"),
                    "direction": parsed.get("direction", "None"),
                    "analysis_text": parsed.get("analysis_text", "Technical analysis was generated successfully."),
                    "model_used": "gpt-4.1-mini (Euri)"
                }
            else:
                last_error = f"Euri API returned status code {response.status_code}: {response.text}"
                print(last_error)
        except Exception as e:
            last_error = f"Euri API request failed: {str(e)}"
            print(last_error)
    else:
        last_error = "EURI_API_KEY is not defined in the environment."
        print(last_error)
        
    # --- Fallback to Groq Cascade list ---
    if GROQ_API_KEY:
        groq_headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Cascade list of robust models on Groq
        models = [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
        
        for model_name in models:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            
            try:
                print(f"Falling back: sending request for {symbol} to Groq API using model {model_name}...")
                response = requests.post(GROQ_URL, headers=groq_headers, json=payload, timeout=20)
                if response.status_code == 200:
                    resp_json = response.json()
                    content = resp_json['choices'][0]['message']['content'].strip()
                    parsed = _parse_ai_json(content)
                    
                    return {
                        "pattern_name": parsed.get("pattern_name", "None"),
                        "confidence": parsed.get("confidence", "None"),
                        "direction": parsed.get("direction", "None"),
                        "analysis_text": parsed.get("analysis_text", "Technical analysis was generated successfully."),
                        "model_used": f"{model_name} (Groq)"
                    }
                else:
                    last_error = f"Model {model_name} returned status code {response.status_code}: {response.text}"
                    print(last_error)
            except Exception as e:
                last_error = f"Model {model_name} failed: {str(e)}"
                print(last_error)
    else:
        if not last_error:
            last_error = "GROQ_API_KEY is missing from the environment credentials."
            
    # Fallback error structure if all options fail
    return {
        "pattern_name": "Error",
        "confidence": "None",
        "direction": "None",
        "analysis_text": f"AI Pattern Recognition engine experienced connectivity problems. Details: {last_error}",
        "model_used": "None"
    }
