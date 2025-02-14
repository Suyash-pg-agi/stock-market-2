import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bson import ObjectId
from typing import List, Optional
import pymongo
import requests
import os
from dotenv import load_dotenv
import json
import uvicorn
from datetime import datetime, timedelta, date, timezone
from bson.errors import InvalidId
from fastapi.responses import JSONResponse
import logging
import pandas as pd
import concurrent.futures
from polygon import RESTClient
import re
import json


# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Fetch credentials from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
MONGO_DB_HOST = os.getenv("MONGO_DB_HOST")
MONGO_DB_PORT = int(os.getenv("MONGO_DB_PORT", 27017))
MONGO_DB_USER = os.getenv("MONGO_DB_USER")
MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")


# MongoDB Connection
mongo_client = pymongo.MongoClient(
    host=MONGO_DB_HOST,
    port=MONGO_DB_PORT,
    username=MONGO_DB_USER,
    password=MONGO_DB_PASSWORD
)

client = RESTClient(api_key=POLYGON_API_KEY)

# Connect to the ima-user database
db = mongo_client["ima-user"]
users_collection = db["users"]
portfolios_collection = db["portfolios"]
portfolio_stocks_collection = db["portfolio_stocks"]
user_wallets = db["wallets"]

logging.basicConfig(level=logging.DEBUG) 
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

load_dotenv()
polygon_key = os.getenv("POLYGON_API_KEY")
client = RESTClient(api_key=polygon_key)


# Import necessary modules at the top
from pydantic import BaseModel, Field
from typing import List, Optional

# Define Pydantic models for structured AI responses
class TrendAnalysis(BaseModel):
    trend_description: str
    best_month: str
    worst_month: str

class HistoricalAnalysis(BaseModel):
    current_price: Optional[float]
    average_price: Optional[float]
    volatility: Optional[float]

class RiskAnalysis(BaseModel):
    volatility: Optional[str]
    trend: str

class StockRecommendation(BaseModel):
    stock_ticker: str
    stock_name: str
    sector: str
    recommendation_rationale: str
    potential_growth_stability_income: str
    trend_analysis: TrendAnalysis
    historical_analysis: HistoricalAnalysis
    risk_analysis: RiskAnalysis
    why_best_for_you: str
    predicted_price_change_percent: str

class RefinedStockRecommendationsOutput(BaseModel):
    refinedRecommendations: List[StockRecommendation]

def call_llama_api(system_message, user_message, max_new_tokens=500):
    try:
        response = requests.post(
            url="http://103.160.106.17:3000/generate",
            json={
                "system_message": system_message,
                "user_message": user_message,
                "max_new_tokens": max_new_tokens
            }
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLaMA API: {str(e)}")
    
def call_struct_llama_api(system_message, user_message, struct, max_new_tokens):
    try:
        response = requests.post(
            url="http://103.160.106.17:3000/generate_structured",
            json={
                "system_message": system_message,
                "user_message": user_message,
                "struct": struct,
                "max_new_tokens": max_new_tokens
            }
        )
        response.raise_for_status()
        data = response.json()
        # logging.debug("DEBUG call_struct_llama_api raw data: %r", data)
        return data
    except requests.RequestException as e:
        logging.error(f"Error communicating with LLaMA API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with LLaMA API: {str(e)}")


# def fetch_current_price(ticker: str):
#     url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         if 'results' in data and 'p' in data['results']:
#             return data['results']['p']
#     return None


# Helper function to validate ObjectId
def get_valid_objectid(user_id: str):
    try:
        return ObjectId(user_id)
    except Exception:
        raise ValueError("Invalid user ID format. Please provide a valid ObjectId.")

# Helper functions for MongoDB queries
def fetch_user_data(user_id: ObjectId):
    user = users_collection.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user

def fetch_portfolios(user_id: ObjectId):
    return list(portfolios_collection.find({"userId": user_id}))

def fetch_wallet(user_id: ObjectId):
    return list(user_wallets.find({"userId": user_id}))

def fetch_portfolio_stocks(portfolio_id: ObjectId):
    return list(portfolio_stocks_collection.find({"portfolioId": portfolio_id}))

def fetch_company_data(ticker: str):
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}


def fetch_stock_trends(ticker: str):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # Fetch data for the last 90 days
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    return []

def analyze_risk(stock_trends: List[dict], stock_info: dict):
    if not stock_trends:
        return "Unknown"
    price_changes = [abs(trend["c"] - trend["o"]) for trend in stock_trends]
    avg_change = sum(price_changes) / len(price_changes)
    beta = stock_info.get("beta", "N/A")
    pe_ratio = stock_info.get("peRatio", "N/A")
    risk_level = "Low Risk"
    if avg_change > 5 or (beta != "N/A" and beta > 1.2) or (pe_ratio != "N/A" and pe_ratio > 25):
        risk_level = "High Risk"
    elif avg_change > 2 or (beta != "N/A" and beta > 1.0):
        risk_level = "Medium Risk"
    return risk_level


def get_historical_data_summary(stocks):
    historical_data_summary = ""
    for stock in stocks:
        ticker = stock.get("ticker")
        stock_info = fetch_company_data(ticker)
        stock_trends = fetch_stock_trends(ticker)
        current_price = fetch_current_price(ticker)
        risk = analyze_risk(stock_trends, stock_info)
        stock["sector"] = stock_info.get("sector")
        stock["risk"] = risk
        stock["current_price"] = current_price
        historical_data_summary += f"\nTicker: {ticker}\nRisk: {risk}\nCurrent Price: {current_price}\n"
        historical_data_summary += "\n".join([
            f"- {datetime.fromtimestamp(trend['t']/1000).strftime('%Y-%m-%d')}: Open={trend['o']}, Close={trend['c']}"
            for trend in stock_trends[:3]
        ])
    return historical_data_summary


def extract_percentage(percentage_str):
    match = re.search(r"([\+\-]?[\d\.]+)%", percentage_str)
    if match:
        return float(match.group(1))
    return None

def extract_numeric_value(price_str):
    matches = re.findall(r"[\d\.]+", price_str.replace(',', ''))
    if matches:
        return float(matches[0])
    return None

def calculate_potential_upside(ticker: str):
    stock_trends = fetch_stock_trends(ticker)
    if not stock_trends:
        return None
    # Calculate average daily percentage change over the last 90 days
    daily_changes = [
        (trend['c'] - trend['o']) / trend['o'] * 100
        for trend in stock_trends if trend['o'] != 0
    ]
    if not daily_changes:
        return None
    average_daily_change = sum(daily_changes) / len(daily_changes)
    # Limit the potential upside to a realistic range
    MAX_DAILY_MOVEMENT_PERCENT = 5.0
    potential_upside = min(max(average_daily_change, -MAX_DAILY_MOVEMENT_PERCENT), MAX_DAILY_MOVEMENT_PERCENT)
    return potential_upside
    
def generate_suggestions_with_llama(portfolio_name: str, stocks: List[dict], user_preferences: List[dict], historical_data_summary: str):
    # Filter and sort stocks based on relevance
    filtered_stocks = [
        stock for stock in stocks if stock.get("current_price") is not None
    ]
    filtered_stocks = sorted(filtered_stocks, key=lambda s: abs(s["current_price"]), reverse=True)[:10]

    stock_list = "\n".join([
        f"{stock['ticker']} - Sector: {stock.get('sector', 'Unknown')}, Risk: {stock.get('risk')}, Current Price: {stock.get('current_price')}"
        for stock in filtered_stocks
    ])
    preferences_text = "\n".join([f"{q['question']}: {q['answer']}" for q in user_preferences])
    system_message = f"""Portfolio '{portfolio_name}' positions:
{stock_list}
Analysis:
- Historical & risk assessment: {historical_data_summary}
- Investment preferences: {preferences_text}

Recommend 3 distinct stocks or strategies to enhance the portfolio and provide 24-hour predictions for current positions.
If the input positions are fewer than 3, describe the existing stocks instead.

Guidelines:
1. Diversification: Ensure balanced sector exposure.
2. Risk Management: Evaluate risk-reward and volatility.
3. Market Trends: Use recent earnings, macroeconomic indicators, and sector data.
4. Liquidity & Costs: Focus on liquid trades with low transaction costs.
5. Time Horizon: Align with the user's investment period.
6. Unique Data: Ensure each suggestion and prediction provides unique data for different stocks. Do not repeat data about the same stock.

Output (each suggestion and prediction must have a unique, distinct single value) - This is the output structure:
class StockSuggestion():
    ticker: str
    sector: str
    risk: str
    rationale: str
    growthPotential: Optional[str]
    historicalAnalysis: str
    targetPrice_for_24h: float

class StockPrediction():
    ticker: str
    predictedPrice_for_24h: float
    confidence: str
    potentialChange: str

class StockMarketOutput():
    suggestions: List[StockSuggestion]
    predictions_for_24h: List[StockPrediction]
"""
    user_message = """Recommend 3 additional stocks or trading strategies to align with the user's goals, improve portfolio performance, and provide 24-hour predictions for their current stocks.
    If the given input stock details are less than 3, explain the existing stocks instead of repeating them. Ensure each suggestion and prediction provides unique data for different stocks."""

    try:
        suggestions_output = call_struct_llama_api(system_message, user_message, "StockMarketOutput", 1000)
        # print("DEBUG after call_struct_llama_api:", suggestions_output)
        
        if isinstance(suggestions_output, str):
            try:
                suggestions_output = json.loads(suggestions_output)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Server returned invalid JSON: {e}"
                )
        suggestions = suggestions_output.get("suggestions", [])
        for suggestion in suggestions:
            ticker = suggestion.get("ticker")
            # Use the potential upside calculated earlier
            potential_upside_percent = calculate_potential_upside(ticker)
            # Fetch current price
            current_price = fetch_current_price(ticker)
            if current_price is not None and potential_upside_percent is not None:
                # Compute targetPrice_for_24h
                target_price = current_price * (1 + potential_upside_percent / 100)
                suggestion["currentPrice"] = f"{current_price:.2f}"
                suggestion["targetPrice_for_24h"] = f"{target_price:.2f}"
                # Remove 'predictedPrice_for_24h' if present
                if 'predictedPrice_for_24h' in suggestion:
                    del suggestion['predictedPrice_for_24h']
                # Format targetPercent with '+' or '-' sign
                if potential_upside_percent >= 0:
                    target_percent_str = f"+{potential_upside_percent:.2f}"
                else:
                    target_percent_str = f"{potential_upside_percent:.2f}"
                suggestion["targetPercent"] = target_percent_str
                # Include potentialUpside
                suggestion["potentialUpside"] = f"{potential_upside_percent:.2f}%"
            else:
                # Handle missing data
                suggestion["currentPrice"] = None
                suggestion["targetPrice_for_24h"] = None
                suggestion["targetPercent"] = None
                suggestion["potentialUpside"] = None
                # Remove 'predictedPrice_for_24h' if present
                if 'predictedPrice_for_24h' in suggestion:
                    del suggestion['predictedPrice_for_24h']
    
        # Similarly process predictions
        predictions = suggestions_output.get("predictions_for_24h", [])
        for prediction in predictions:
            ticker = prediction.get("ticker")
            potential_change_percent = calculate_potential_upside(ticker)
            current_price = fetch_current_price(ticker)
            if current_price is not None and potential_change_percent is not None:
                predicted_price = current_price * (1 + potential_change_percent / 100)
                prediction["currentPrice"] = f"{current_price:.2f}"
                prediction["predictedPrice_for_24h"] = f"{predicted_price:.2f}"
                if potential_change_percent >= 0:
                    percentage_str = f"+{potential_change_percent:.2f}"
                else:
                    percentage_str = f"{potential_change_percent:.2f}"
                prediction["percentage"] = percentage_str
                prediction["potentialChange"] = percentage_str + "%"
            else:
                prediction["currentPrice"] = None
                prediction["predictedPrice_for_24h"] = None
                prediction["percentage"] = None
                prediction["potentialChange"] = None
            # Remove 'targetPrice_for_24h' if present
            if 'targetPrice_for_24h' in prediction:
                del prediction['targetPrice_for_24h']
    
        return suggestions_output
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing GPT response: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {e}")

def convert_volatility(volatility_str):
    """Convert volatility string (e.g., 'High') to a numerical representation."""
    mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Low-Volatility-Risk": 1,
        "Medium-Volatility-Risk": 2,
        "High-Volatility-Risk": 3
    }
    return mapping.get(volatility_str, None)

def sanitize_percentage_change(percent_change):
    """Sanitize the percentage change to be within -5% to +5%."""
    try:
        value = float(percent_change.strip('%'))
        if value > 5:
            return 5.0
        elif value < -5:
            return -5.0
        return value
    except:
        return 0.0

def process_ai_response(validated_response, stock_prices):
    """Process AI response by computing target prices based on percentage changes."""
    processed_recommendations = []
    for rec in validated_response['refinedRecommendations']:
        ticker = rec['stock_ticker']
        current_price = stock_prices.get(ticker, None)
        if current_price is None:
            logging.warning(f"No current price found for ticker: {ticker}")
            continue

        predicted_change = rec.get('predicted_price_change_percent', '0%')
        predicted_change = sanitize_percentage_change(predicted_change)

        target_price = current_price * (1 + predicted_change / 100)
        rec['targetPrice_for_24h'] = round(target_price, 2)
        processed_recommendations.append(rec)

    return {"refinedRecommendations": processed_recommendations}

def validate_recommendations(recommendations, real_data):
    """Validate recommendations to ensure target prices are realistic."""
    validated_recommendations = []
    for rec in recommendations:
        ticker = rec['stock_ticker']
        real_price = real_data.get(ticker, None)
        if real_price is None:
            logging.warning(f"Real price not found for ticker: {ticker}")
            continue

        # Example validation: Ensure target price is within +/-5% of current price
        target_price = rec.get('targetPrice_for_24h', 0)
        lower_bound = real_price * 0.95
        upper_bound = real_price * 1.05
        if not (lower_bound <= target_price <= upper_bound):
            logging.warning(f"Target price for {ticker} is out of realistic range.")
            continue

        validated_recommendations.append(rec)

    return validated_recommendations

# refine the stocks to remove duplicates and adjust the volume
def refine_stocks(stocks):
    unique_stocks = {}

    for stock in stocks:
        ticker = stock.get("ticker")
        if ticker not in unique_stocks:
            unique_stocks[ticker] = stock
        else:
            existing_stock = unique_stocks[ticker]
            if existing_stock.get("transactionType") == "buy" and stock.get("transactionType") == "sell":
                existing_stock["volume"] -= stock.get("volume", 0)
            elif existing_stock.get("transactionType") == "sell" and stock.get("transactionType") == "buy":
                existing_stock["volume"] -= stock.get("volume", 0)
    
    # refined_stocks = {ticker: stock for ticker, stock in unique_stocks.items() if stock.get("volume", 0) != 0}
    return unique_stocks

def analyze_stocks(stock_data, buy_price):
    df = pd.DataFrame(stock_data)
    df['date'] = pd.to_datetime(df['t'], unit='ms')
    current_price = df.iloc[-1]['c']
    
    # 1. Price Change (%)
    price_change_percent = ((current_price - buy_price) / buy_price) * 100
    
    # 2. VWAP Comparison
    current_vwap = df.iloc[-1]['vw']
    vwap_above = bool(current_price > current_vwap)
    
    # 3. Momentum (Rate of Change)
    roc = ((df['c'].iloc[-1] - df['c'].iloc[-10]) / df['c'].iloc[-10]) * 100 if len(df) >= 10 else 0
    
    # 4. Volume Trend
    avg_volume = df['v'].mean()
    last_volume = df.iloc[-1]['v']
    volume_trend = 1 if last_volume > avg_volume else -1
    
    # 5. Maximum Drawdown (%)
    highest_price = df['h'].max()
    lowest_price = df['l'].min()
    max_drawdown_percent = ((highest_price - lowest_price) / highest_price) * 100
    
    # Normalize and weight each metric
    sell_score = (
        0.4 * (price_change_percent / 100) +  # Weight 40%
        0.2 * (roc / 100) +                  # Weight 20%
        0.1 * volume_trend +                 # Weight 10%
        0.2 * (1 if vwap_above else -1) +    # Weight 20%
        -0.1 * (max_drawdown_percent / 100)  # Weight -10%
    )
    
    # Final summary
    summary = {
        "buy_price": round(buy_price, 2),
        "current_price": round(current_price, 2),
        "price_change_percent": round(price_change_percent, 2),
        "sell_score": round(sell_score, 2),
    }
    
    return summary

def format_date(date_obj):
    return date_obj.strftime("%Y-%m-%d")

def current_status(ticker, stocks):
    buy_date_str = stocks.get("date")
    if buy_date_str:
        buy_date = format_date(buy_date_str)
    else:
        buy_date = (datetime.today() - timedelta(days=31)).strftime("%Y-%m-%d")
        stocks['price'] = None

    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{buy_date}/{yesterday}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            data = data["results"]
            if not buy_date_str:
                stocks['price'] = data[0].get('c')
        else:
            return None
    else:
        print(f"Error fetching data for {ticker}: {response.status_code}")
        return None
    
    return analyze_stocks(data, stocks.get("price"))

def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.dropna().empty else 50

def analyze_stock_data(stock_data, ticker_name):
    if not stock_data:
        return {}
    df = pd.DataFrame(stock_data)
    df["return"] = df["c"].pct_change()
    
    # Basic stats
    opening = df.iloc[0]["o"]
    closing = df.iloc[-1]["c"]
    max_h = df["h"].max()
    min_l = df["l"].min()
    sum_volumes = df["v"].sum()
    sum_vw_times_volumes = (df["v"] * df.get("vw", 0)).sum()
    average_vwap = sum_vw_times_volumes / sum_volumes if sum_volumes else 0

    # New metrics
    daily_returns = df["return"].dropna()
    historical_volatility = daily_returns.std() * math.sqrt(252) if not daily_returns.empty else 0
    short_ma = df["c"].rolling(5).mean().iloc[-1] if len(df) >= 5 else closing
    long_ma = df["c"].rolling(20).mean().iloc[-1] if len(df) >= 20 else closing
    trend_strength = (short_ma - long_ma) / (long_ma + 1e-9)
    
    # Simplified RSI
    rsi_value = compute_rsi(df["c"]) if len(df) > 14 else 50

    price_changes = df["c"] - df["o"]
    avg_return = price_changes.mean() if not price_changes.empty else 0
    std_dev = price_changes.std() if price_changes.size > 1 else 0
    sharpe_ratio = avg_return / std_dev if std_dev else 0
    # average_price = df["c"].mean() if not df["c"].empty else 0

    return {
        "ticker": ticker_name,
        "price_change": closing - opening,
        "percent_change": ((closing - opening) / opening) * 100 if opening else 0,
        "price_range": max_h - min_l,
        "average_vwap": average_vwap,
        "total_volume": sum_volumes,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_h - df["c"].min(),
        "rsi": rsi_value,
        "historical_volatility": historical_volatility,
        "trend_strength": trend_strength,
    }

def fetch_aggregates(ticker, mul="1", timespan="day", last_30_days = str((datetime.today() - timedelta(days=31)).strftime("%Y-%m-%d")), yesterday=str((datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"))):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
        f"/range/{mul}/{timespan}/{last_30_days}/{yesterday}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={polygon_key}"
    )
    try:
        response = requests.Session().get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException:
        return []

def get_sell_suggestions(stocks):
    refined_stocks = refine_stocks(stocks)
    analysis_data = {}

    def analyze_stock(ticker, stock):
        # Fetch and analyze the last 30 days' data
        last_30_days_analysis = analyze_stock_data(
            fetch_aggregates(ticker, mul="1", timespan="day", last_30_days = str((datetime.today() - timedelta(days=31)).strftime("%Y-%m-%d")), yesterday=str((datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"))), ticker
        )
        # Keep only relevant keys for last_30_days_analysis
        last_30_days_filtered = {
            "percent_change": round(last_30_days_analysis.get("percent_change", 0), 2),
            "sharpe_ratio": round(last_30_days_analysis.get("sharpe_ratio", 0), 2),
            "rsi": round(last_30_days_analysis.get("rsi", 0), 2),
            "trend_strength": round(last_30_days_analysis.get("trend_strength", 0), 2),
        }

        # Fetch and analyze the data from the buy date
        analysis_from_buy = current_status(ticker, stock)
        return ticker, {
            "Name": fetch_ticker_name(ticker),
            "last_30_days_analysis": last_30_days_filtered,
            "analysis_from_buy": analysis_from_buy,
        }

    # Use multithreading to speed up the analysis for multiple stocks
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_stock, stock.get("ticker"), stock): stock for stock in refined_stocks.values()}
        for future in concurrent.futures.as_completed(futures):
            ticker, result = future.result()
            analysis_data[ticker] = result

    # Sort the analysis_data based on sell_score in descending order and take the first 10 stocks
    sorted_10_analysis = dict(sorted(
        analysis_data.items(),
        key=lambda item: item[1]["analysis_from_buy"]["sell_score"],
        reverse=True
    )[:10])

    system_prompt = f"""**Constraints:**
- Include only stocks with strong reasons to sell.
- Use concise, actionable language for the "Reason".
- Prioritize using the sell score (estimated based on factors like price_change_percent, ROC, volume_trend, VWAP_above, and max_drawdown_percent), but do not include it in the explanation.

**Output format for each stock:**
- **Stock_ticker**: Stock symbol.
- **Name**: Stock’s name or ticker.
- **Reason**: Why selling is recommended.
- **Sell_priority_rank**: Based on urgency, derived from metrics like sell score, performance, or oversold status.
- **Current_price**: Current market price.
- **bought_price**: User bought price.
- **Performance_summary**: Summary of recent and long-term performance.

Here is the list of analyzed stocks the user has invested in:
{sorted_10_analysis}"""

    user_prompt = """Analyze the given list of stocks and identify 3 stocks best suited to sell. Return the output in prioritized order (most urgent to sell first), providing clear reasoning for each stock."""

    if len(sorted_10_analysis) <= 3: # If less than 3 stocks, return all
        system_prompt = "You are a financial assistant specializing in stock analysis. Analyze the given list of stocks. Prioritize the output (most urgent to sell first) and provide clear reasoning for each recommendation. Do not add multiple values. Provide exactly the same number of output values as given in the input values." + system_prompt
        user_prompt = "Analyze the given list of stocks. Return the output in prioritized order (most urgent to sell first), providing clear reasoning for each stock. Do not add multiple values. Provide exactly the same number of output values as given in the input values."
        return call_struct_llama_api(
            system_prompt,
            user_prompt,
            "SellStockList",
            max_new_tokens=1000
        ).get("recommendations", "None")

    system_prompt = "You are a financial assistant specializing in stock analysis. Analyze the given list of stocks and identify 3 stocks best suited to sell. Prioritize the output (most urgent to sell first) and provide clear reasoning for each recommendation."+ system_prompt
    return call_struct_llama_api(
        system_prompt,
        user_prompt,
        "SellStockList",
        max_new_tokens=1000
    ).get("recommendations", "None")

# API Endpoint to fetch portfolio suggestions
@app.get("/portfolio/suggestions/", tags=["Portfolio"])
def get_portfolio_suggestions(user_id: str):
    try:
        user_id_obj = get_valid_objectid(user_id)
        # logging.debug("user_id_obj: ", str(user_id_obj))

        user = fetch_user_data(user_id_obj)
        # logging.debug("user_data: ", str(user))

        portfolios = fetch_portfolios(user_id_obj)
        # logging.debug("portfolios: ", str(portfolios))

        if not portfolios:
            raise HTTPException(status_code=404, detail="No portfolios found for the user.")
        response_data = []
        for portfolio in portfolios:
            portfolio_id = portfolio["_id"]
            stocks = fetch_portfolio_stocks(portfolio_id)
            if not stocks:
                continue
            # Generate historical data summary and update stocks
            historical_data_summary = get_historical_data_summary(stocks)
            # logging.debug("historical_data_summary: ", historical_data_summary)

            # Prepare stock data for GPT
            stock_data = [{
                "ticker": stock.get("ticker"),
                "sector": stock.get("sector"),
                "risk": stock.get("risk"),
                "current_price": stock.get("current_price")
            } for stock in stocks]
            suggestions = generate_suggestions_with_llama(
                portfolio["name"], stock_data, user.get("questions", []), historical_data_summary
            )
            # logging.debug("suggestions: ", suggestions)
            response_data.append({
                "portfolio": portfolio["name"],
                "suggestions": suggestions["suggestions"],
                "sell_suggestions": get_sell_suggestions(stocks)
            })
        return {"data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/predictions/", tags=["Portfolio"])
def get_portfolio_predictions(user_id: str):
    try:
        user_id_obj = get_valid_objectid(user_id)
        user = fetch_user_data(user_id_obj)
        portfolios = fetch_portfolios(user_id_obj)
        if not portfolios:
            raise HTTPException(status_code=404, detail="No portfolios found for the user.")
        response_data = []
        for portfolio in portfolios:
            portfolio_id = portfolio["_id"]
            stocks = fetch_portfolio_stocks(portfolio_id)
            logging.debug(f"stocks: {stocks}")
            if not stocks:
                continue
            # Generate historical data summary and update stocks
            historical_data_summary = get_historical_data_summary(stocks)
            # Prepare stock data for GPT
            stock_data = [{
                "ticker": stock.get("ticker"),
                "sector": stock.get("sector"),
                "risk": stock.get("risk"),
                "current_price": stock.get("current_price")
            } for stock in stocks]
            suggestions = generate_suggestions_with_llama(
                portfolio["name"], stock_data, user.get("questions", []), historical_data_summary
            )
            response_data.append({
                "portfolio": portfolio["name"],
                "predictions_for_24h": suggestions["predictions_for_24h"],
                "sell_predictions": get_sell_suggestions(stocks)
            })
        return {"data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RecommendationRequest(BaseModel):
    user_id: str

db = mongo_client['ima-user']
collection = db['users']

def get_user_by_id2(user_id: str):
    """Fetch user information from MongoDB using the user's _id."""
    try:

        user = collection.find_one({"_id": ObjectId(user_id)})
        if user:
            return user 
        else:
            print("User not found.")
            return None
    except Exception as e:
        print(f"Error fetching user by ID: {e}")
        return None


def get_user_questions(user_id):
    """Fetch user-specific questions and answers from MongoDB by _id."""
    user = get_user_by_id2(user_id)  
    if user and 'questions' in user:
        structured_data = [{"question": q["question"], "answer": q["answer"]} for q in user['questions']]
        # print("User Questions",structured_data)
        return structured_data
    return None


def fetch_current_price(ticker: str):
    url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        price = data.get('results', {}).get('p')
        if price is not None:
            logging.debug(f"Fetched price for {ticker}: {price}")
            return price
        logging.warning(f"No price data found for ticker: {ticker}")
    except requests.RequestException as e:
        logging.error(f"Error fetching price for {ticker}: {str(e)}")
    return None

def fetch_tickers_by_market(market_type="stocks", locale="us"):
    """Fetch a list of tickers dynamically from Polygon API based on market type and locale."""
    url = f"https://api.polygon.io/v3/reference/tickers?market={market_type}&locale={locale}&active=true&limit=50&sort=ticker&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Check if 'results' exists and is a list
        if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
            logging.debug(f"Sample fetched tickers: {data['results'][:5]}")
            
            # Filter and validate each entry in results
            validated_tickers = [
                {
                    "ticker": stock.get("ticker"),
                    "name": stock.get("name", "N/A"),
                    "market": stock.get("market", "N/A"),
                    "locale": stock.get("locale", "N/A")
                }
                for stock in data['results']
                if isinstance(stock, dict) and "ticker" in stock
            ]
            
            if validated_tickers:
                return validated_tickers
            else:
                logging.warning("No valid tickers found after filtering.")
        else:
            logging.warning("Unexpected format for 'results' in response data.")

    except requests.RequestException as e:
        logging.error(f"Error fetching ticker data: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error while processing ticker data: {str(e)}")
    
    return [] 


def analyze_sentiments(user_preferences):
    """Analyze sentiment of user preferences"""
    system_prompt = f"""You are an AI assistant analyzing user investment preferences. Based on the following user responses:

    {json.dumps(user_preferences, indent=2)}

    Analyze the user's overall sentiment and preferences. Provide a **short summary** covering:
    1. Overall sentiment (e.g., Positive, Neutral, Negative).
    2. Key traits (e.g., High risk, Short-term focus, Income focus, Long-term focus).
    3. Sector preferences (e.g., Technology, Healthcare, Finance, Other).
    4. Volatility tolerance (e.g., High, Medium, Low).

    Summarize in **3-4 sentences** and return the result in plain text."""
    user_prompt = "Summarize the overall sentiment of the user and classify their investment preferences concisely in 3-4 sentences"
    
    sentiments_analysis = call_llama_api(system_prompt, user_prompt, max_new_tokens=500)
    
    return sentiments_analysis


def analyze_risk2(ticker):
    """Analyze stock risk using Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2024-10-31?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            high_prices = [entry["h"] for entry in data["results"]]
            low_prices = [entry["l"] for entry in data["results"]]

            volatility = max(high_prices) - min(low_prices)
            trend = "Going Up" if high_prices[-1] > low_prices[0] else "Going Down"

            return {
                "volatility": round(volatility, 2),
                "trend": trend
            }
        else:
            return {"volatility": None, "trend": "Unknown"}
    else:
        return {"volatility": None, "trend": "Unknown"}


def fetch_historical_data(ticker, start_date, end_date):
    """Fetch historical ticker data from the Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/month/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"]
        else:
            # print(f"No historical data found for {ticker}.")
            return None
    else:
        print(f"Error fetching historical data for {ticker}: {response.status_code}")
        return None

def analyze_historical_data(historical_data):
    """Analyze historical data to derive insights."""
    if not historical_data:
        return {
            # "current_price": None,
            "average_price": None,
            # "volatility": None
        }

    prices = [entry["vw"] for entry in historical_data]
    high_prices = [entry["h"] for entry in historical_data]
    low_prices = [entry["l"] for entry in historical_data]

    avg_price = sum(prices) / len(prices) if prices else None
    volatility = max(high_prices) - min(low_prices) if high_prices and low_prices else None

    current_price = prices[-1] if prices else None

    return {
        # "current_price": round(current_price, 2) if current_price else None,
        "average_price": round(avg_price, 2) if avg_price else None,
        # "volatility": round(volatility, 2) if volatility else None
    }

def fetch_ticker_name(ticker_name):
    """Fetch top gainers from Polygon API."""
    url = f"https://api.polygon.io/v3/reference/tickers?ticker={ticker_name}&active=true&limit=100&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['results'] == []:
            return "N/A"
        return str(data.get("results", [])[0].get("name", ""))
    except requests.RequestException as e:
        return "N/A"

def fetch_top_gainers_losers(type="gainers"):
    """Fetch top gainers from Polygon API."""
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/{type}?apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # The structure of the response might vary; typically, it contains a list of gainers under 'tickers'
        tickers = data.get("tickers", [])
        
        # Extract relevant information
        validated_tickers = [
            {
                "ticker": stock.get("ticker"),
                "name": fetch_ticker_name(stock.get('ticker')),
                "market": "stocks",
                "locale": "us"
            }
            for stock in tickers
            if isinstance(stock, dict) and "ticker" in stock
        ]
        
        if validated_tickers:
            logging.info(f"Fetched top gainers/losers: {[t['ticker'] for t in validated_tickers]}")
            return validated_tickers
        else:
            logging.warning("No valid top gainers found.")
            return []
    
    except requests.RequestException as e:
        logging.error(f"Error fetching top gainers: {str(e)}")
        return []


def analyze_trend(historical_data):
    """
    Provide detailed trend analysis with month-by-month percentage changes.
    
    Args:
    historical_data (list): Historical price data from Polygon API
    
    Returns:
    dict: Trend analysis with specific monthly performance insights
    """
    if not historical_data or len(historical_data) < 2:
        return {
            "trend_description": "Insufficient data for trend analysis",
            "monthly_trends": []
        }
    
    # month-to-month percentage changes
    monthly_trends = []
    for i in range(1, len(historical_data)):
        prev_price = historical_data[i-1]['vw']
        current_price = historical_data[i]['vw']
        
        percent_change = ((current_price - prev_price) / prev_price) * 100
        monthly_trends.append({
            "month": i,
            "percent_change": round(percent_change, 2)
        })
    
    # Aggregate 
    avg_monthly_change = sum(trend['percent_change'] for trend in monthly_trends) / len(monthly_trends)
    
    #  overall trend description
    if avg_monthly_change > 5:
        trend_description = f"Strong upward trend with average monthly growth of {round(avg_monthly_change, 2)}%"
    elif avg_monthly_change > 0:
        trend_description = f"Moderate upward trend, averaging {round(avg_monthly_change, 2)}% monthly increase"
    elif avg_monthly_change == 0:
        trend_description = "Stable with minimal price fluctuations"
    elif avg_monthly_change > -5:
        trend_description = f"Moderate downward trend, averaging {round(abs(avg_monthly_change), 2)}% monthly decline"
    else:
        trend_description = f"Significant downward trend with average monthly decline of {round(abs(avg_monthly_change), 2)}%"
    
    return {
        "trend_description": trend_description,
        # "monthly_trends": monthly_trends,
        "best_month": max(monthly_trends, key=lambda x: x['percent_change']),
        "worst_month": min(monthly_trends, key=lambda x: x['percent_change'])
    }

def fetch_final_recommendations(sentiments_analysis, tickers_data):
    recommendations = []
    analysis_data = {}

    end_date = date.today()
    start_date = end_date - timedelta(days=180)

    def analyze_ticker(ticker_data):
        if not isinstance(ticker_data, dict) or "ticker" not in ticker_data:
            logging.warning(f"Skipping invalid ticker_data: {ticker_data}")
            return None

        ticker = ticker_data.get("ticker")
        name = ticker_data.get("name", "Unknown")
        sector = ticker_data.get("market", "Unknown")

        historical_data = fetch_historical_data(ticker, str(start_date), str(end_date))
        trend_analysis = analyze_trend(historical_data)
        historical_analysis = analyze_historical_data(historical_data)
        risk_data = analyze_risk2(ticker)
        current_price = fetch_current_price(ticker)

        return {
            "ticker": ticker,
            "name": name,
            "current_price": current_price,
            "historical_analysis": historical_analysis,
            "trend_analysis": trend_analysis,
            "risk_analysis": risk_data,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_ticker, ticker_data): ticker_data for ticker_data in tickers_data}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                recommendations.append(result)

    refined_recommendations = [
        rec for rec in recommendations if isinstance(rec, dict)
        and rec.get("risk_analysis", {}).get("volatility") is not None
    ]

    system_prompt = f"""
    As an expert stock market analyst, refine the stock recommendations to align with user preferences and market conditions.
 
    ---
    **Rules for Recommendations:**

    1. **Diversification:**
    - Avoid over-concentration in any  single stock, sector, or strategy.

    2. **Risk Management:**
    - Evaluate the risk-reward user preferences of each suggestion.
    - Consider historical volatility and establish clear stop-loss levels.

    3. **Market Trends:**
    - Integrate insights from recent market data such as earnings reports, sector trends, macroeconomic indicators (e.g., GDP, employment rates), and geopolitical developments.

    4. **Liquidity and Transaction Costs:**
    - Propose trades with high liquidity to ensure seamless execution.
    - Minimize transaction costs by focusing on options with tight bid-ask spreads.

    5. **Time Horizon Alignment:**
    - Match stock recommendations with the user’s specified trading horizon:
        - **Short-Term:** Focus on momentum and high-volatility stocks.
        - **Medium-Term:** Emphasize growth potential and solid fundamentals.
        - **Long-Term:** Prioritize stability, dividends, and blue-chip stocks.

    **Execution Guidelines:**
    - Always justify recommendations with data-backed reasoning.
    - Ensure that your output is tailored to the user's prefences , specified risk appetite and investment objectives.

    ---

    User Sentiment Analysis:
    {sentiments_analysis}

    Please suggest three refined recommendations in with detailed reasoning from the below top 20 companies.
    {json.dumps(refined_recommendations, indent=None, separators=(',', ':'))}
    """
    user_prompt = "Based on your preferences and market conditions, provide three refined stock recommendations to align with user's investment goals."

    output = call_struct_llama_api(
        system_prompt,
        user_prompt,
        "RefinedStockRecommendationsOutput",
        max_new_tokens=500
    )

    # If output is a string, parse into dict
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Server returned invalid JSON: {e}"
            )

    return output 

def remove_deduplicate_stocks(stocks):
    unique_stocks = {}
    for stock in stocks:
        ticker = stock.get('ticker')
        if ticker and ticker not in unique_stocks:
            unique_stocks[ticker] = stock
        else:
            # Optionally, update with latest data or merge information
            pass
    return list(unique_stocks.values())

def remove_portfolio_stocks(tickers, user_id):
    user_id_obj = get_valid_objectid(user_id)
    user = fetch_user_data(user_id_obj)
    portfolios = fetch_portfolios(user_id_obj)

    if not portfolios:
        raise HTTPException(status_code=404, detail="No portfolios found for the user.")
    all_user_stocks = []

    for portfolio in portfolios:
        portfolio_id = portfolio["_id"]
        stocks = fetch_portfolio_stocks(portfolio_id)
        all_user_stocks.extend(stock.get("ticker") for stock in stocks)

    return [ticker for ticker in tickers if ticker not in all_user_stocks]


@app.post("/recommend-user-stocks", tags=["Stock Recommendations"])
async def recommend_stocks(request: RecommendationRequest):
    user_id = request.user_id
    try:
        # Fetch user data
        user = get_user_by_id2(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch user questions/preferences
        user_preferences = get_user_questions(user_id)
        if not user_preferences:
            raise HTTPException(status_code=404, detail="User preferences not found")
        # logging.debug(f"User preferences: {user_preferences}")

        # Perform sentiment analysis
        sentiments_analysis = analyze_sentiments(user_preferences)
        tickers_data = fetch_top_gainers_losers()

        # Remove user's portfolio stocks from recommendations
        tickers_data = remove_portfolio_stocks(tickers_data, user_id)

        # Deduplicate tickers to prevent redundant processing
        tickers_data = remove_deduplicate_stocks(tickers_data)[:10]
        # logging.debug("tickers_data: " + str(tickers_data))

        # Fetch final recommendations based on sentiments and tickers
        final_data = fetch_final_recommendations(sentiments_analysis, tickers_data)
        # logging.debug("DEBUG fetch_final_recommendations output:", final_data)
        final_recommendations = final_data.get("refined_recommendations") or final_data.get("refinedRecommendations")

        if not final_recommendations:
            raise HTTPException(status_code=500, detail="No 'refinedRecommendations' found")

        tickers_data = fetch_top_gainers_losers(type="losers")
        tickers_data = remove_deduplicate_stocks(tickers_data)[:10]
        logging.debug("tickers_data: " + str(tickers_data))

        return JSONResponse(content={
            "user_id": user_id,
            "user_name": user.get("name", "Unknown"),
            "recommendations": final_recommendations,
            "sell_recommendations": get_sell_suggestions(tickers_data)
        })

    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except Exception as e:
        logging.error(f"Error in /recommend-user-stocks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

def all_active_tickers(client):
    days_back = 1
    while True:
        date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        data = list(client.get_grouped_daily_aggs(date=date))
        if data:
            return list(data)
        days_back += 1

# Function to filter tickers based on user balance
def stock_by_balance(balance, client):
    all_tickers = all_active_tickers(client)
    affordable_tickers = [ticker for ticker in all_tickers if ticker.close <= balance]
    return affordable_tickers

# Function to rank affordable tickers based on the order in tickers_analysis.json
def rank_affordable_tickers(tickers_analysis, affordable_tickers):
    affordable_ticker_dict = {ticker.ticker: ticker for ticker in affordable_tickers}
    ranked_tickers = [affordable_ticker_dict[ticker['ticker']] for ticker in tickers_analysis if ticker['ticker'] in affordable_ticker_dict]
    return ranked_tickers

@app.get("/balance_stock_suggestions", tags=["Stock Suggestions"])
async def balance_stock_suggestions(user_id: str):
    try:
        # Replace these with your actual functions to fetch user data and client
        user_id_obj = get_valid_objectid(user_id)
        wallet = fetch_wallet(user_id_obj)
        if not wallet:
            raise HTTPException(status_code=404, detail="Wallet not found.")
        balance = wallet[0].get("balance", 0)

        # Load the JSON analysis data
        with open('tickers_analysis.json', 'r') as file:
            tickers_analysis = json.load(file)

        # Get affordable tickers
        affordable_tickers = stock_by_balance(balance, client)

        # Rank affordable tickers based on their order in tickers_analysis.json
        ranked_tickers = rank_affordable_tickers(tickers_analysis, affordable_tickers)

        return {"user_balance": balance, "data": ranked_tickers[:10]}

    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Tickers analysis file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def call_balance_suggestions(user_id):
    base_url = "http://103.160.106.17:8000"
    url = f"{base_url}/balance_stock_suggestions"

    response = requests.get(url, params={"user_id": user_id})
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Error: {response.status_code}, {response.text}"

@app.get("/check_balance_change", tags=["Check Balance"])
def check_balance_change():
    try:
        # MongoDB Connection
        mongo_client = pymongo.MongoClient(
            host=MONGO_DB_HOST,
            port=MONGO_DB_PORT,
            username=MONGO_DB_USER,
            password=MONGO_DB_PASSWORD
        )

        # Connect to the ima-user database
        db = mongo_client["ima-user"]
        user_wallets = db["wallets"]

        file_path = "wallets.json"
        updated_user_ids = []
        current_wallets = list(user_wallets.find({}))
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, 'r') as f:
                existing_wallets = json.load(f)
            
            existing_userIds = set(existing_wallets.keys())

            for wallet in current_wallets:
                if str(wallet.get('userId')) not in existing_userIds:
                    existing_wallets[str(wallet.get('userId'))] = wallet.get('balance')
                if wallet.get('balance') != existing_wallets.get(str(wallet.get('userId'))):
                    updated_user_ids.append(str(wallet.get('userId')))
                    existing_wallets[str(wallet.get('userId'))] = wallet.get('balance')

            with open(file_path, 'w') as f:
                json.dump(existing_wallets, f)
        else:
            with open(file_path, 'w') as f:
                existing_wallets = {}
                for wallet in current_wallets:
                    existing_wallets[str(wallet.get('userId'))] = wallet.get('balance')
                json.dump(existing_wallets, f)
                updated_user_ids = list(existing_wallets.keys())

        user_suggestions = {}
        for user_id in updated_user_ids:
            user_suggestions[user_id] = call_balance_suggestions(user_id)

        # print("user_suggestions: ", user_suggestions)
        
        return {"data": user_suggestions}

    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Tickers analysis file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

find_queryType_prompt = """Based on the user query, determine the nature of the request and classify it into one of three categories:

1. Single Company Analysis:
    - If the query explicitly mentions a specific stock ticker or company name, or requests detailed performance analysis, trends, or insights for that single company, set:
        singleComapanyQuestion: True
        RelatedCompany: False
        StockExchangeQuestion: False
        OverallComapaniesQuestion: False
        GeneralQuestion: False

2. Related Companies: 
    - If the query is about asking trends for the related companies, or its competitors of the given company, set:
        singleComapanyQuestion: False
        RelatedCompany: True
        StockExchangeQuestion: False
        OverallComapaniesQuestion: False
        GeneralQuestion: False

3. Top Companies Analysis:
    - If the query specifically asks for top companies ranking or requests to identify top companies (e.g., "top stocks", "best performing companies", "ranking of top companies"), set:
        singleComapanyQuestion: False
        RelatedCompany: False
        StockExchangeQuestion: False
        OverallComapaniesQuestion: True
        GeneralQuestion: False

4. Stock Exchange Analysis:
    - If the query is about fetching the list of companies listed or query related to a specific stock exchange such as the New York Stock Exchange (NYSE), set:
        singleComapanyQuestion: False
        RelatedCompany: False
        StockExchangeQuestion: True
        OverallComapaniesQuestion: False
        GeneralQuestion: False

5. General Market Inquiry:
    - For other queries about broader market trends, investment strategies, or general stock market insights where no specific company or ranking is clearly requested, set:
        singleComapanyQuestion: False
        RelatedCompany: False
        StockExchangeQuestion: False
        OverallComapaniesQuestion: False
        GeneralQuestion: True
"""

find_aggs_prompt = """
Your task is to select the most suitable parameters based on the question context. Follow these guidelines:
    - Use `from_date` and `to_date` based on the question context:
    - If analyzing data for a single day, set `from_date` to the previous day and `to_date` to the current date. Use `timespan` as `"hour"`.
    - For ranges of 2–7 days, set `timespan` as `"hour"`.
    - For ranges greater than 7 days, use `timespan` as `"day"`.
    {
        "ticker": "AAPL",
        "multiplier": 1,
        "timespan": "day",
        "from_date": "2023-01-09",
        "to_date": "2023-02-10",
    }
"""

find_ticker_param_prompt = """Your task is to select the most suitable parameters based on the question context. Follow these guidelines:
This api is used to get the overall top few companies sorted based on multiple parameters.
    - Your task is to find the correct parameters based on the user query and state then as 'True' to be used and rest as 'False'
    - In case of general question, then provide all the parameters as 'True' value
    - Parameter list: ticker, price_change, percent_change, price_range, average_price, average_vwap, total_volume, sharpe_ratio, max_drawdown, rsi, historical_volatility, trend_strength
    - Example: Which stocks currently are in upward trend?
        Output: 
        {
            "price_change" = True
            "percent_change" = True
            "price_range" = False
            "average_vwap" = True
            "total_volume" = False
            "sharpe_ratio" = False
            "max_drawdown" = False
            "rsi" = True
            "historical_volatility" = True
            "trend_strength" = True
        }
"""

find_ticker_name = """Your task is to find the ticker name of the company based on this given data.
Make sure you provide the company ticker, not the company name.
Example: Based on the provided data, the current trend in Apple stock market appears to be a moderate upward trend.
Output: AAPL
"""

find_exchange_code = """Your task is to find the stock exchange code based on the user query.
Example user query: "Based on information found through your accessible sources, is there an upward trend in the New York Stock Exchange?"
Output: NYSE
"""

find_exchange_tickers = """Your task is to find the list of top companies listed on the given stock exchange. 
Ensure you provide at least 10 of the most popular companies listed on the given stock exchange.

Example: 
Stock Exchange: NYSE

Output:[
    {"ticker_name": "BRK.A"},
    {"ticker_name": "BRK.B"},
    {"ticker_name": "JPM"},
    {"ticker_name": "JNJ"},
    {"ticker_name": "V"},
    {"ticker_name": "PG"},
    {"ticker_name": "XOM"},
    {"ticker_name": "KO"},
    {"ticker_name": "DIS"},
    {"ticker_name": "MCD"}
]
"""


# Utility functions
def round_or_none(value, precision=2):
    return round(value, precision) if value is not None else None

# Calculations
def calculate_metrics(data, prev_volume=None):
    volume, vw, o, c, h, l, t, n = data['v'], data['vw'], data['o'], data['c'], data['h'], data['l'], data['t'], data['n']
    metrics = {
        "date": datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "price_change": round_or_none(c - o),
        "percent_change": round_or_none(((c - o) / o) * 100),
        "price_range": round_or_none(h - l),
        "relative_volatility": round_or_none((h - l) / vw),
        "average_trade_size": round_or_none(volume / n if n else 0),
        "volume_change": round_or_none(((volume - prev_volume) / prev_volume) * 100) if prev_volume else None,
        "volume": round_or_none(volume),
        "high_price": round_or_none(h),
        "low_price": round_or_none(l),
        "opening_price": round_or_none(o),
        "closing_price": round_or_none(c),
        "number_of_trades": n
    }
    return metrics

def analyze_data(data):
    return [calculate_metrics(data[i], data[i-1]['v'] if i > 0 else None) for i in range(len(data))]

def summarize_analysis(results):
    total_volume = round_or_none(sum(r["volume"] for r in results))
    average_volatility = round_or_none(sum(r["relative_volatility"] for r in results) / len(results))
    average_trade_size = round_or_none(sum(r["average_trade_size"] for r in results) / len(results))
    
    high_prices = [r["high_price"] for r in results if r["high_price"] is not None]
    low_prices = [r["low_price"] for r in results if r["low_price"] is not None]
    percent_changes = [r["percent_change"] for r in results if r["percent_change"] is not None]
    
    # Max high price and min low price (Remove if not significant in your context)
    max_high_price = round_or_none(max(high_prices)) if high_prices else None
    min_low_price = round_or_none(min(low_prices)) if low_prices else None
    
    # Total trades
    total_trades = sum(r["number_of_trades"] for r in results)
    
    # Average percent change (only if percentage changes exist)
    average_percent_change = round_or_none(sum(percent_changes) / len(percent_changes)) if percent_changes else None
    
    # Build concise summary
    return {
        "total_volume": total_volume,
        "average_volatility": average_volatility,
        "average_trade_size": average_trade_size,
        "max_high_price": max_high_price,
        "min_low_price": min_low_price,
        "total_number_of_trades": total_trades,
        "average_percent_change": average_percent_change
    }

# Function to calculate the multiplier
def cal_multiplier(current_mul, agg_len):
    if (agg_len < 15 and current_mul == 1) or (15 <= agg_len <= 20):
        return current_mul
    if agg_len < 15 and current_mul > 1:
        return current_mul - 1
    thresholds = [
        (50, 4), (100, 6), (150, 9), (200, 13), (300, 19), (400, 28), (500, 32), (600, 40), (700, 48), (800, 64), (900, 78), (1000, 72), (1250, 88), (1500, 115), (1750, 135), (2000, 145), (2500, 185), (3000, 215), (3500, 245), (4000, 285), (4500, 320), (5001, 370)
    ]
    
    for threshold, increment in thresholds:
        if agg_len < threshold:
            return current_mul + increment
    return current_mul

# # Function to fetch aggregate data
# def fetch_aggregates(ticker, mul, timespan, from_date, to_date, polygon_key):
#     url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mul}/{timespan}/{from_date}/{to_date}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_key}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         return data.get('results', [])
#     return []

# Function to handle `get_aggs` API logic
def handle_get_aggs(api_data, polygon_key):
    ticker = api_data.get("ticker")
    mul = api_data.get("multiplier", 1)
    timespan = api_data.get("timespan", "day")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    past_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    from_date = api_data.get("from_date", past_month)
    to_date = api_data.get("to_date", yesterday)

    if mul == None: mul = 1

    visited_mul = set()

    aggs = []
    while True:
        aggs = fetch_aggregates(ticker, mul, timespan, from_date, to_date)
        cal_mul = cal_multiplier(mul, len(aggs))
        if cal_mul in visited_mul:
            break
        visited_mul.add(cal_mul)
        if mul == cal_mul:
            break
        mul = cal_mul

    if not aggs:
        print("Incomplete data provided! Fetching with a multiplier of 1.")
        aggs = fetch_aggregates(ticker, 1, "day", past_month, yesterday)[:20]

    analysis = analyze_data(aggs)
    summary = summarize_analysis(analysis)

    return "Aggregate Analysis: " + " ".join([str(agg) for agg in analysis]) + ". Summary: ".join([str(summary)])

# Calculate the overall score for each company
def calculate_overall_score(percentiles, selected_columns, weights=None):
    if weights is None:
        weights = {col: 1 for col in selected_columns}
    normalized_weights = pd.Series(weights) / sum(weights.values())
    percentiles["overall_score"] = percentiles[selected_columns].mul(normalized_weights[selected_columns]).sum(axis=1)
    return percentiles

# Get the top 15 companies based on the overall score
def get_top15Companies(selected_columns=["price_change"]):
    with open('tickers_analysis.json', 'r') as file:
        data = json.load(file)

    percentile_list = [item['percentile_value'] for item in data]
    actual_values_list = [item['actual_values'] for item in data]

    percentiles_df = pd.DataFrame(percentile_list)
    actual_values_df = pd.DataFrame(actual_values_list)

    result = calculate_overall_score(percentiles_df, selected_columns)

    tickers = [item['ticker'] for item in data]
    result.insert(0, 'ticker', tickers)
    result = result.sort_values(by="overall_score", ascending=False)
    result = result[['ticker', 'overall_score']]

    ranked_companies = result.merge(actual_values_df, on='ticker')
    top_15_companies = ranked_companies.head(15).to_dict(orient='records')

    for company in top_15_companies:
        for key in company:
            if isinstance(company[key], (int, float)):
                company[key] = round(company[key], 2)

    return top_15_companies

# Function to handle `handle_overall_data` API logic
def handle_overall(api_data):
    selected_columns = [field for field, value in api_data.items() if value]
    top_com_data = get_top15Companies(selected_columns)
    return "Top 15 Companies: " + " ".join([str(company) for company in top_com_data])

# def get_ticker_news(ticker, polygon_key):
#     url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&order=asc&limit=10&apiKey={polygon_key}"
#     response = requests.get(url)
#     all_news = []
#     if response.status_code == 200:
#         data = response.json()
#         for news in data.get('results', []):
#             all_news.append(news.get('insights'))
#     return []

def get_related_companies(ticker, polygon_key):
    url = f"https://api.polygon.io/v1/related-companies/{ticker}?apiKey={polygon_key}"
    response = requests.get(url)
    related_com_list = []
    if response.status_code == 200:
        data = response.json().get("results")
        for company in data:
            related_com_list.append(company.get("ticker"))
        return related_com_list
    else:
        return None
    
def llama_struct_explaination(system_message, user_message, struct, max_new_tokens = 100):
    try:
        response = requests.post(
            url="http://103.160.106.17:3000/generate_structured",
            json={
                "system_message": system_message,
                "user_message": user_message,
                "struct": struct,
                "max_new_tokens": max_new_tokens
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLaMA API: {str(e)}")
    
def llama_explaination(user_query, polygon_output, max_new_tokens= 1000):
    url = "http://103.160.106.17:3000/generate"
    system_message = 'Analyze the following stock market data and answer the question in short. Focus on key trends, price movements, volatility, volume fluctuations, and any significant insights drawn from the data. Include only information relevant to the specific question: ' + str(polygon_output)
    text_input = {
        "system_message": system_message,
        "user_message": user_query,
        "max_new_tokens": max_new_tokens
    }
    try:
        response = requests.post(url, json=text_input)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Error details: {response.text}")
    except Exception as e:
        return f"An error occurred: {e}"

CHAT_SESSSION = {}

def add_conv_history(token_id, user_query, response):
    if token_id not in CHAT_SESSSION:
        CHAT_SESSSION[token_id] = []
    CHAT_SESSSION[token_id].append({"user_query": user_query, "response": response})
    if len(CHAT_SESSSION[token_id]) > 3:
        CHAT_SESSSION[token_id] = CHAT_SESSSION[token_id][-3:]
    if len(CHAT_SESSSION) > 100000:
        CHAT_SESSSION.pop(0)

@app.get("/chat", tags=["Chat"])
def chatbot(user_query: str, token_id: str):
    if token_id not in CHAT_SESSSION:
        CHAT_SESSSION[token_id] = []
    previous_messages = str(CHAT_SESSSION[token_id]) if len(CHAT_SESSSION[token_id]) >= 1 else ""
    # print("previous_messages: ", previous_messages)
    try:
        if user_query.lower() == "exit":
            return "You have successfully exited the chatbot session. Thank you for using the chatbot."
        
        queryType = llama_struct_explaination(find_queryType_prompt, user_query, "FindQueryType")
        if queryType.get("singleComapanyQuestion") == True:        
            today_date = datetime.now().strftime('%Y-%m-%d')
            model_gen_api = llama_struct_explaination(find_aggs_prompt  + ". Previous Chat: " + previous_messages, user_query + f" Today's Date= '{today_date}'", "APIDetails")
            print("model_func: ", model_gen_api)

            start_date = model_gen_api.get("from_date", "")
            end_date = model_gen_api.get("to_date", "")
            dates = 'Below data is between the dates: ' + start_date + " and " + end_date

            try:
                polygon_output = handle_get_aggs(model_gen_api, polygon_key)
            except Exception as error:
                return (f"error: Invalid given data!!\nerror: {error}")

            if not polygon_output:
                return ("No Stock market data found!!. Please try again")

            # with open('polygon_output.txt', 'w') as file:
            #     file.write(polygon_output)

            system_message = (
                "You are a professional financial data analyst. Analyze the provided stock market data with precision. "
                "Focus on key trends such as price movements (highest, lowest, opening, and closing values), "
                "volatility, volume fluctuations, and significant patterns over the specified period. "
                "Answer concisely and directly, including only insights relevant to the user's question."
            )
            llama_output = llama_explaination(
                system_message + dates + ". Stock Market data: " + str(polygon_output) + ". Previous Chat: " + previous_messages,
                user_query
            ).get("response")

            add_conv_history(token_id, user_query, llama_output)

            return llama_output
        
        elif queryType.get("OverallComapaniesQuestion") == True:
            model_gen_api = llama_struct_explaination(find_ticker_param_prompt + ". Previous Chat: " + previous_messages, user_query, "TickerParams")
            print("model_func: ", model_gen_api)

            try:
                polygon_output = handle_overall(model_gen_api)
            except Exception as error:
                return (f"error: Invalid given data!!\nerror: {error}")

            if not polygon_output:
                return ("No Stock market data found!!. Please try again")

            with open('polygon_output.txt', 'w') as file:
                file.write(str(polygon_output))

            yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            last_30_days = (datetime.today() - timedelta(days=31)).strftime("%Y-%m-%d")
            dates = 'Below data is of past 30 days between dates: ' + last_30_days + " and " + yesterday

            system_message = (
                "You are a professional financial data analyst. Analyze the provided stock market data with precision. "
                "Here is a data of top 5 companies ranked on user's query. "
                "Focus on key trends such as price movements, volatility, volume fluctuations, and significant patterns over the specified period. "
                "Answer concisely and directly, including only insights relevant to the user's question."
            )

            llama_output = llama_explaination(
                system_message + dates + ". Stock Market data: " + str(polygon_output) + ". Previous Chat: " + previous_messages,
                user_query
            ).get("response")

            add_conv_history(token_id, user_query, llama_output)

            return llama_output
        
        elif queryType.get("StockExchangeQuestion") == True:
            model_gen_api = llama_struct_explaination(find_exchange_code + ". Previous Chat: " + previous_messages, user_query, "GetExchangeCode")
            print("model_func: ", model_gen_api)
            
            company_list = llama_struct_explaination(find_exchange_tickers, model_gen_api.get("exchange_code"), "GetExchangeTickers")
            company_summaries = []
            week_back = (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            for ticker_entry in company_list.get("exchange_tickers", []):
                ticker = ticker_entry.get("ticker_name")
                aggs = fetch_aggregates(str(ticker), "6", 'hour', week_back, yesterday)
                if not aggs or len(aggs) == 0:
                    continue
                analysis = analyze_data(aggs)
                summary = summarize_analysis(analysis)
                summary["ticker"] = ticker
                company_summaries.append(summary)
                break

            system_message = (
                "You are a professional financial data analyst. Given data is the analysis of the companies of the provided stock exchange. "
                "Focus on key trends such as price movements, volatility, volume fluctuations, and significant patterns over the specified period. "
                "Here is the analysis of companies of the provided stock exchange: "
            ) + " ".join([str(company) for company in company_summaries])

            llama_output = llama_explaination(
                system_message + ". Previous Chat: " + previous_messages,
                user_query
            ).get("response")

            add_conv_history(token_id, user_query, llama_output)

            return llama_output
        
        elif queryType.get("RelatedCompany") == True:
            model_gen_api = llama_struct_explaination(find_ticker_name + ". Previous Chat: " + previous_messages, user_query, "GetTickerName")
            print("model_func: ", model_gen_api)
            related_companies = get_related_companies(model_gen_api.get("ticker_name"), polygon_key)
            company_summaries = []
            week_back = (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            for ticker in related_companies or []:
                aggs = fetch_aggregates(str(ticker), "6", 'hour', week_back, yesterday)
                if not aggs or len(aggs) == 0:
                    continue
                analysis = analyze_data(aggs)
                summary = summarize_analysis(analysis)
                summary["ticker"] = ticker
                company_summaries.append(summary)

            system_message = (
                "You are a professional financial data analyst. Given data is the analysis of related companies of the provided company. "
                "Focus on key trends such as price movements, volatility, volume fluctuations, and significant patterns over the specified period. "
                "Here is the analysis of related companies: "
            ) + " ".join([str(company) for company in company_summaries])

            llama_output = llama_explaination(
                system_message + ". Previous Chat: " + previous_messages,
                user_query
            ).get("response")

            add_conv_history(token_id, user_query, llama_output)

            return llama_output
        
        else:
            print("General Output Selected")
            system_message = (
                "You are a financial data analyst. Based on your knowledge, answer concisely and directly, including only insights relevant to the user's question."
            )
            llama_output = llama_explaination(
                system_message + ". Previous Chat: " + previous_messages,
                user_query
            ).get("response")

            add_conv_history(token_id, user_query, llama_output)

            return llama_output
    except Exception as e:
        print(f"Please try again. An error occurred: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="103.160.106.17", port=8000) 
