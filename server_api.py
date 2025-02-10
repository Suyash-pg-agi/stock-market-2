from datetime import datetime, timedelta
import json
import logging
import math
import os
import threading
import time
from typing import List, Optional
import concurrent
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
import requests
import schedule
from transformers import pipeline
import torch
from ollama import chat
from pydantic import ValidationError
from polygon import RESTClient
from scipy.stats import rankdata
import numpy as np
from keras import models, layers, callbacks
from sklearn.preprocessing import MinMaxScaler
import uvicorn


load_dotenv()
polygon_key = os.getenv("POLYGON_API_KEY")
client = RESTClient(api_key=polygon_key)


yesterday = str((datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"))
last_30_days = str((datetime.today() - timedelta(days=31)).strftime("%Y-%m-%d"))

def fetch_aggregates(ticker, session, mul="1", timespan="day", from_date=last_30_days, to_date=yesterday):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
        f"/range/{mul}/{timespan}/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={polygon_key}"
    )
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException:
        return []
    
def all_active_tickers(client):
    return [data for data in client.get_grouped_daily_aggs(date=yesterday)]

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

def calculate_percentiles_with_adjustments(data):
    percentiles = data.copy()
    reverse_columns = [
        "max_drawdown",
        "rsi",
    ]
    for column in percentiles.columns:
        if column == "ticker" or percentiles[column].nunique() <= 1:
            continue
        rank = rankdata(percentiles[column], method="average") / len(percentiles[column])
        percentiles[column] = (
            1 - rank
            if column in reverse_columns
            else rank
        )

    return percentiles.fillna(0)

def calculate_overall_score(percentiles, weights=None):
    if weights is None:
        weights = {col: 1 for col in percentiles.columns if col != "ticker"}
    normalized_weights = pd.Series(weights) / sum(weights.values())
    percentiles["overall_score"] = percentiles.drop(columns=["ticker"]).mul(normalized_weights).sum(axis=1)
    return percentiles

def rank_companies(data, weights=None):
    percentiles = calculate_percentiles_with_adjustments(data)
    ranked_data = calculate_overall_score(percentiles, weights)
    return ranked_data.sort_values(by="overall_score", ascending=False)

def get_ranked_companies(ticker_list: List[str], mul="1", timespan="day", from_date=last_30_days, to_date=yesterday):
    ticker_names = list({agg for agg in ticker_list})

    with requests.Session() as session, concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        analysis_results = list(executor.map(
            lambda ticker: analyze_stock_data(fetch_aggregates(ticker, session, mul, timespan, from_date, to_date), ticker),
            ticker_names
        ))

    df = pd.DataFrame([res for res in analysis_results if res]).dropna()
    top_companies = rank_companies(df)

    data_out = []
    analysis_dict = {item["ticker"]: item for item in analysis_results if item}
    for _, row in top_companies.iterrows():
        ticker = row["ticker"]
        data_out.append({
            "ticker": ticker,
            "actual_values": analysis_dict.get(ticker, {}),
            "percentile_value": row.drop(["ticker", "overall_score"]).to_dict()
        })
    
    return data_out

def update_top_companies_file():
    all_tickers = all_active_tickers(client)
    ticker_names = list({agg.ticker for agg in all_tickers})
    data_out = get_ranked_companies(ticker_names)
    filename = f"tickers_analysis.json"
    with open(filename, "w") as f:
        json.dump(data_out, f, indent=4, default=str)

    print("Top Companies Ranking Updated in the file:", filename)


# Load the Llama model
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Initialize FastAPI app
app = FastAPI()

# Define request model
class RequestData(BaseModel):
    system_message: str
    user_message: str
    max_new_tokens: int = 500

@app.post("/generate")
def generate_response(data: RequestData):
    try:
        system_message = f"Please keep your response short and concise, only 2-3 lines. {data.system_message}. Optionally, if it fits the context, you may conclude by asking a follow-up starting with 'would you like to know more about' to invite further curiosity about the topic, but only include such a follow-up when it feels natural."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": data.user_message},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=data.max_new_tokens,
        )
        # Correct extraction of generated_text
        response = outputs[0]["generated_text"][2]["content"]
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    
class StringOutput(BaseModel):
    output: str

class FunctionStruct(BaseModel):
    function_name: str
    arguments: dict

class APIDetails(BaseModel):
    ticker: Optional[str] = None
    multiplier: Optional[int] = None
    timespan: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None

class StockSuggestion(BaseModel):
    ticker: str
    sector: str
    risk: str
    rationale: str
    growthPotential: Optional[str]
    historicalAnalysis: str
    targetPrice_for_24h: float

class StockPrediction(BaseModel):
    ticker: str
    predictedPrice_for_24h: float
    confidence: str
    potentialChange: str

class StockMarketOutput(BaseModel):
    suggestions: List[StockSuggestion]
    predictions_for_24h: List[StockPrediction]

class QuestionSentiment(BaseModel):
    question: str
    sentiment: str

class UserPreferencesAnalysis(BaseModel):
    highRisk: bool
    shortTermFocus: bool
    incomeFocus: bool
    longTermFocus: bool
    sectorPreference: str
    volatilityTolerance: str

class InvestmentPreferencesOutput(BaseModel):
    sentimentAnalysis: List[QuestionSentiment]
    userPreferences: UserPreferencesAnalysis

class RiskAnalysis(BaseModel):
    volatility: Optional[str]
    trend: str

class TrendAnalysis(BaseModel):
    trend_description: str
    best_month: str
    worst_month: str

class HistoricalAnalysis(BaseModel):
    performance: str
    current_price: float

class StockRecommendation(BaseModel):
    ticker: str
    sector: str
    risk: str
    rationale: str
    targetPrice_for_24h: Optional[float]
    marketTrends: str
    liquidityConsideration: str
    volatilityRisk: str
    stock_name: str
    trend_analysis: TrendAnalysis
    historical_analysis: HistoricalAnalysis
    risk_analysis: RiskAnalysis
    predicted_price: Optional[float]
    suitability_for_user: str

class RefinedStockRecommendationsOutput(BaseModel):
    refinedRecommendations: List[StockRecommendation]

class FindQueryType(BaseModel):
    singleComapanyQuestion: bool
    RelatedCompany: bool
    StockExchangeQuestion: bool
    OverallComapaniesQuestion: bool
    GeneralQuestion: bool

class TickerParams(BaseModel):
    price_change : bool
    percent_change : bool
    price_range : bool
    average_vwap : bool
    total_volume : bool
    sharpe_ratio : bool
    max_drawdown : bool
    rsi : bool
    historical_volatility : bool
    trend_strength : bool

class SellStockRecommendation(BaseModel):
    Stock_ticker: str
    Name: str
    Reason: str
    Sell_priority_rank: int
    Current_price: float
    bought_price:float
    Performance_summary: str

class SellStockList(BaseModel):
    recommendations: List[SellStockRecommendation]

class PredictStockPrice(BaseModel):
    predicted_15_minutes: float
    predicted_1_hour: float
    predicted_1_day: float
    predicted_1_week: float

class StructuredResponse(BaseModel):
    system_message: str
    user_message: str
    struct: str
    max_new_tokens: int = 1000

class GetTickerName(BaseModel):
    ticker_name: str

class GetExchangeCode(BaseModel):
    exchange_code: str

class GetExchangeTickers(BaseModel):
    exchange_tickers: List[GetTickerName]

class PredictionRequest(BaseModel):
    ticker: str
    timespan_option: str

class_mapping = {"RefinedStockRecommendationsOutput":RefinedStockRecommendationsOutput,
                 "InvestmentPreferencesOutput":InvestmentPreferencesOutput,
                 "StockMarketOutput":StockMarketOutput,
                 "FunctionStruct":FunctionStruct,
                 "StringOutput":StringOutput,
                 "APIDetails": APIDetails,
                 "FindQueryType": FindQueryType,
                 "TickerParams": TickerParams,
                 "SellStockList": SellStockList,
                 "PredictStockPrice": PredictStockPrice,
                 "GetTickerName": GetTickerName,
                 "GetExchangeTickers": GetExchangeTickers,
                 "GetExchangeCode": GetExchangeCode
                }

@app.post("/generate_structured")
def llama_structured_response(request: StructuredResponse):
    struct = class_mapping.get(request.struct)
    if not struct:
        raise HTTPException(status_code=400, detail="Invalid struct type provided.")
    
    try:
        response = chat(
            messages=[
                {"role": "system", "content": request.system_message},
                {"role": "user", "content": request.user_message},
            ],
            model='llama3.1:8b-instruct-fp16',
            format=struct.model_json_schema(),
        )
        
        # Validate and parse the response using Pydantic
        validated_response = struct.model_validate_json(response.message.content)
        return validated_response.model_dump()
    except ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"Invalid response structure: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating structured response: {str(e)}")
    
# Function to dynamically set MULTIPLIER and TIMESPAN
def get_timespan_settings(timespan_option):
    """
    Assigns multiplier and timespan based on user selection.
    """
    if timespan_option == "day":
        return 1, "day"
    elif timespan_option == "hour":
        return 1, "hour"
    elif timespan_option == "5min":
        return 5, "minute" 
    else:
        raise ValueError("Invalid timespan option. Choose from 'day', 'hour', or 'minute'.")
    
# Create a single session for all HTTP requests
session = requests.Session()

def fetch_current_price(ticker: str):
    url = f"https://api.polygon.io/v2/last/trade/{ticker.upper()}?apiKey={polygon_key}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        price = data.get('results', {}).get('p')
        if price is not None:
            return price
    except requests.RequestException as e:
        return None

def fetch_aggs(multiplier, timespan, TICKER, from_date, to_date):
    if not from_date or not to_date:
        raise ValueError("Invalid date format. Ensure 'from_date' and 'to_date' are in YYYY-MM-DD format.")
    url = (f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/"
           f"{multiplier}/{timespan}/{from_date}/{to_date}?limit=50000&apiKey={polygon_key}")
    response = session.get(url)
    data = response.json()
    # if 'results' not in data or len(data['results']) < 80:
    #     raise Exception("Not enough data fetched")
    # print("API Response:", data)
    # print("API Response:", url)


    if 'results' not in data:
        raise ValueError(f"API Error: 'results' key missing. Response: {data}")
    
    return data['results']

def fetch_data_until_threshold(current_date, timespan_option, TICKER, threshold=500, backfill_days=30):
    data = []
    MULTIPLIER, TIMESPAN = get_timespan_settings(timespan_option)

    while len(data) <= threshold:
        from_date = (current_date - pd.DateOffset(days=backfill_days)).strftime('%Y-%m-%d')
        to_date = current_date.strftime('%Y-%m-%d')
        # print(f"Fetching data from {from_date} to {to_date} for {TICKER} ({TIMESPAN})")
        fetched_data = fetch_aggs(MULTIPLIER, TIMESPAN, TICKER, from_date, to_date)
        if fetched_data:
            data.extend(fetched_data)
        else:
            print(f"Warning: No data fetched for {TICKER} from {from_date} to {to_date}")
        data.extend(fetched_data)
        current_date = current_date - pd.DateOffset(days=backfill_days)
        # print(f"Fetched {len(fetched_data)} data points. Total data points: {len(data)}")
        
    return data, MULTIPLIER, TIMESPAN

# Removing outliers
def remove_outliers(df, column="c"):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def sma(series, length):
    """Simple Moving Average (SMA)"""
    return series.rolling(window=length, min_periods=1).mean()

def ema(series, length):
    """Exponential Moving Average (EMA)"""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def vwap(high, low, close, volume):
    """Volume Weighted Average Price (VWAP)"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def obv(close, volume):
    """On-Balance Volume (OBV)"""
    obv_values = [0]  # Initialize OBV with 0
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv_values.append(obv_values[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv_values.append(obv_values[-1] - volume.iloc[i])
        else:
            obv_values.append(obv_values[-1])  # No change in OBV

    return pd.Series(obv_values, index=close.index)

def macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence (MACD)"""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'hist': macd_histogram})

def bollinger_bands(series, length=20, std=2):
    """Bollinger Bands (BBANDS)"""
    sma_series = sma(series, length)
    rolling_std = series.rolling(window=length, min_periods=1).std()
    upper_band = sma_series + (rolling_std * std)
    lower_band = sma_series - (rolling_std * std)
    return pd.DataFrame({'bb_upper': upper_band, 'bb_middle': sma_series, 'bb_lower': lower_band})

def remove_outliers(df, column, threshold=3):
    """Remove outliers using Z-score"""
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[(z_scores.abs() < threshold)]

def preprocess_data(data):
    df = pd.DataFrame(data)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    df = df[~df.index.duplicated()].sort_index()

    available_features = ['o', 'h', 'l', 'c', 'n', 'v', 'vw']
    features_to_use = [col for col in available_features if col in df.columns]
    df = df[features_to_use]

    df['ma20'] = sma(df['c'], length=20)
    df['rsi'] = rsi(df['c'], length=14)
    df['ema50'] = ema(df['c'], length=50)
    df['vwap'] = vwap(df['h'], df['l'], df['c'], df['v'])
    df['obv'] = obv(df['c'], df['v'])

    macd_df = macd(df['c'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd_df], axis=1)

    bbands_df = bollinger_bands(df['c'], length=20, std=2)
    df = pd.concat([df, bbands_df], axis=1)

    df.dropna(inplace=True)
    df = remove_outliers(df, "c")

    return df

def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

def build_optimized_lstm_model(input_shape):
    model = models.Sequential([
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='swish'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, validation_split=0.2, epochs=30, batch_size=64):
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stop, reduce_lr]
    )
    return history

def predict_next_value(model, features_scaled, target_scaler, seq_length):
    # Use the last 'seq_length' values from the features as the input sequence for prediction
    last_sequence = features_scaled[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)  # shape: (1, seq_length, num_features)
    next_pred_scaled = model.predict(last_sequence)
    next_pred = target_scaler.inverse_transform(next_pred_scaled)
    return next_pred[0, 0]

def predict_next_price(timespan_option, TICKER):
    current_date = pd.Timestamp.now()
    data, MULTIPLIER, TIMESPAN = fetch_data_until_threshold(current_date, timespan_option, TICKER)
    
    logging.debug("processing")
    df = preprocess_data(data)
    logging.debug(df.__len__)

    features = df.copy()
    target = df[['c']]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)

    SEQ_LENGTH = 20
    X, y = create_sequences(features_scaled, target_scaled, SEQ_LENGTH)

    # Use 100% of the data for training
    # print(f"Training LSTM model for {TIMESPAN} data...")    
    model = build_optimized_lstm_model((SEQ_LENGTH, features.shape[1]))
    train_model(model, X, y)

    return predict_next_value(model, features_scaled, target_scaler, SEQ_LENGTH)

@app.post("/predict_price")
def predict_price(request: PredictionRequest):
    try:
        # Validate timespan option
        valid_timespans = ["day", "hour", "5min"]
        if request.timespan_option not in valid_timespans:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timespan option. Must be one of: {valid_timespans}"
            )
        current_date = pd.Timestamp.now()
        data, MULTIPLIER, TIMESPAN = fetch_data_until_threshold(current_date, request.timespan_option, request.ticker)
        if not data:
            raise HTTPException(status_code=404, detail=f"No data found for {request.ticker} in {request.timespan} timeframe")
        predicted_price = predict_next_price(request.timespan_option, request.ticker)
        
        return {
            "ticker": request.ticker.upper(),
            "timespan": request.timespan_option,
            "current_price": fetch_current_price(request.ticker),
            "predicted_price": float(predicted_price)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

lock = threading.Lock()

# def update_top_companies_file():
#     if lock.locked():
#         return

#     with lock:
#         try:
#             time.sleep(100)
#         except Exception as e:
#             print(f"Error updating companies: {e}")

# def run_scheduler():
#     schedule.every(10).minutes.do(update_top_companies_file)
#     while True:
#         schedule.run_pending()
#         time.sleep(300)

LOG_FILE = "last_run.txt"

def save_last_run():
    """Save the last execution date to a file"""
    with open(LOG_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

def get_last_run():
    """Get the last execution date from the file"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return f.read().strip()
    return None

def check_missed_run():
    """Check if the task was missed and execute it if necessary"""
    last_run = get_last_run()
    today = datetime.now().strftime("%Y-%m-%d")

    if last_run != today:
        print("Missed execution! Running task now...")
        update_top_companies_file()

if __name__ == "__main__":
    # Run the scheduler in a separate thread
    # scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    # scheduler_thread.start()

    schedule.every().day.at("00:00").do(update_top_companies_file)

    uvicorn.run(app, host="103.160.106.17", port=3000)

    # Run check once at startup
    check_missed_run()

    while True:
        schedule.run_pending()
        time.sleep(3600)