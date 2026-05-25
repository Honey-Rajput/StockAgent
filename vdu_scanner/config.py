# config.py
import pytz

# Timezone setting
IST_TIMEZONE = pytz.timezone('Asia/Kolkata')

# Default Algorithmic Thresholds
DRY_VOLUME_THRESHOLD = 0.40    # 40% of baseline = "dry"
MIN_VOLUME_RATIO = 2.0         # 2x surge minimum
MIN_PRICE_CHANGE = 1.5         # 1.5% min price move on breakout day
DRY_ZONE_MIN_DAYS = 30
DRY_ZONE_MAX_DAYS = 60
LOOKBACK_DAYS = 450            # Fetch 450 calendar days to guarantee 310+ clean trading days

# NIFTY 50 Static Tickers
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
    "BHARTIARTL", "SBIN", "LICI", "ITC", "HINDUNILVR", 
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", 
    "ADANIENT", "KOTAKBANK", "TATAMOTORS", "ONGC", "NTPC", 
    "AXISBANK", "ADANIPORTS", "ULTRACEMCO", "TITAN", "COALINDIA", 
    "POWERGRID", "M&M", "ASIANPAINT", "WIPRO", "BAJAJFINSV", 
    "JIOFIN", "JSWSTEEL", "TATASTEEL", "ADANIPOWER", "GRASIM", 
    "LTIM", "NESTLEIND", "BAJAJ-AUTO", "SBILIFE", "HINDALCO", 
    "ADANIGREEN", "HAL", "TECHM", "BRITANNIA", "CIPLA", 
    "TRENT", "INDUSINDBK", "EICHERMOT", "BPCL", "TATACONSUM"
]

# NIFTY 100 Static Tickers
NIFTY100_SYMBOLS = [
    # NIFTY 50 Tickers
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
    "BHARTIARTL", "SBIN", "LICI", "ITC", "HINDUNILVR", 
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", 
    "ADANIENT", "KOTAKBANK", "TATAMOTORS", "ONGC", "NTPC", 
    "AXISBANK", "ADANIPORTS", "ULTRACEMCO", "TITAN", "COALINDIA", 
    "POWERGRID", "M&M", "ASIANPAINT", "WIPRO", "BAJAJFINSV", 
    "JIOFIN", "JSWSTEEL", "TATASTEEL", "ADANIPOWER", "GRASIM", 
    "LTIM", "NESTLEIND", "BAJAJ-AUTO", "SBILIFE", "HINDALCO", 
    "ADANIGREEN", "HAL", "TECHM", "BRITANNIA", "CIPLA", 
    "TRENT", "INDUSINDBK", "EICHERMOT", "BPCL", "TATACONSUM",
    # Additional NIFTY Next 50 Tickers
    "ABB", "AMBUJACEM", "APOLLOHOSP", "DMART", "BAJAJHLDNG", 
    "BANKBARODA", "BEL", "BHEL", "BIOCON", "BOSCHLTD", 
    "CANBK", "CHOLAFIN", "COFORGE", "COLPAL", "CONCOR", 
    "DLF", "DABUR", "DIVISLAB", "DRREDDY", "GAIL", 
    "GMRINFRA", "GODREJCP", "HDFCLIFE", "HEROMOTOCO", "ICICIGI", 
    "ICICIPRULI", "IEX", "IOC", "IRCTC", "IRFC", 
    "INDIGO", "JINDALSTEL", "JUBLFOOD", "LICHSGFIN", "MARICO", 
    "MUTHOOTFIN", "NYKAA", "OFSS", "PIDILITIND", "PIIND", 
    "PNB", "SBICARD", "SRF", "SHREECEM", "SHRIRAMFIN", 
    "SIEMENS", "TATACOMM", "TATAELXSI", "TATAPOWER", "TVSMOTOR", 
    "UNITDSPR", "VBL", "VEDL", "ZOMATO", "ZYDUSLIFE"
]

# Quick mapping dictionary for prominent stocks to speed up loading
COMPANY_NAME_MAP = {
    "RELIANCE": "Reliance Industries Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "INFY": "Infosys Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "BHARTIARTL": "Bharti Airtel Ltd.",
    "SBIN": "State Bank of India",
    "LICI": "Life Insurance Corporation of India",
    "ITC": "ITC Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.",
    "LT": "Larsen & Toubro Ltd.",
    "BAJFINANCE": "Bajaj Finance Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "SUNPHARMA": "Sun Pharmaceutical Industries Ltd.",
    "ADANIENT": "Adani Enterprises Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "TATAMOTORS": "Tata Motors Ltd.",
    "ONGC": "Oil & Natural Gas Corporation Ltd.",
    "NTPC": "NTPC Ltd.",
    "AXISBANK": "Axis Bank Ltd.",
    "ADANIPORTS": "Adani Ports & Special Economic Zone Ltd.",
    "ULTRACEMCO": "UltraTech Cement Ltd.",
    "TITAN": "Titan Company Ltd.",
    "COALINDIA": "Coal India Ltd.",
    "POWERGRID": "Power Grid Corporation of India Ltd.",
    "M&M": "Mahindra & Mahindra Ltd.",
    "ASIANPAINT": "Asian Paints Ltd.",
    "WIPRO": "Wipro Ltd.",
    "BAJAJFINSV": "Bajaj Finserv Ltd.",
    "JIOFIN": "Jio Financial Services Ltd.",
    "JSWSTEEL": "JSW Steel Ltd.",
    "TATASTEEL": "Tata Steel Ltd.",
    "ADANIPOWER": "Adani Power Ltd.",
    "GRASIM": "Grasim Industries Ltd.",
    "LTIM": "LTIMindtree Ltd.",
    "NESTLEIND": "Nestle India Ltd.",
    "BAJAJ-AUTO": "Bajaj Auto Ltd.",
    "SBILIFE": "SBI Life Insurance Company Ltd.",
    "HINDALCO": "Hindalco Industries Ltd.",
    "ADANIGREEN": "Adani Green Energy Ltd.",
    "HAL": "Hindustan Aeronautics Ltd.",
    "TECHM": "Tech Mahindra Ltd.",
    "BRITANNIA": "Britannia Industries Ltd.",
    "CIPLA": "Cipla Ltd.",
    "TRENT": "Trent Ltd.",
    "INDUSINDBK": "IndusInd Bank Ltd.",
    "EICHERMOT": "Eicher Motors Ltd.",
    "BPCL": "Bharat Petroleum Corporation Ltd.",
    "TATACONSUM": "Tata Consumer Products Ltd.",
    "ABB": "ABB India Ltd.",
    "AMBUJACEM": "Ambuja Cements Ltd.",
    "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd.",
    "DMART": "Avenue Supermarts Ltd. (DMart)",
    "BANKBARODA": "Bank of Baroda",
    "BEL": "Bharat Electronics Ltd.",
    "BHEL": "Bharat Heavy Electricals Ltd.",
    "DLF": "DLF Ltd.",
    "DABUR": "Dabur India Ltd.",
    "DRREDDY": "Dr. Reddy's Laboratories Ltd.",
    "GAIL": "GAIL (India) Ltd.",
    "HDFCLIFE": "HDFC Life Insurance Company Ltd.",
    "IOC": "Indian Oil Corporation Ltd.",
    "INDIGO": "InterGlobe Aviation Ltd. (IndiGo)",
    "PNB": "Punjab National Bank",
    "TATAPOWER": "Tata Power Company Ltd.",
    "VEDL": "Vedanta Ltd.",
    "ZOMATO": "Zomato Ltd."
}

def get_company_name(symbol):
    """
    Returns clean company name or falls back to symbol
    """
    base_sym = symbol.replace(".NS", "").upper()
    return COMPANY_NAME_MAP.get(base_sym, f"{base_sym} India")
