import re
import os
import json
import logging
import requests
import yfinance as yf
import spacy
import praw
import sqlite3
import nltk
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import shap
from lime import lime_tabular
import streamlit as st
from difflib import SequenceMatcher

# ---------------------------
# Setup Logging & Environment
# ---------------------------
logging.basicConfig(
    filename="sentiment_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)
load_dotenv()

# ---------------------------
# Reddit & NLP Setup
# ---------------------------
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Persistent SQLite Connection & Tables
# ---------------------------
DB_NAME = "stock_sentiment_v9.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

cursor.executescript('''
CREATE TABLE IF NOT EXISTS stock_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    sentiment_score REAL,
    sentiment TEXT,
    post_content TEXT,
    subreddit TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON stock_sentiment (ticker);
CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON stock_sentiment (timestamp);
CREATE INDEX IF NOT EXISTS idx_sentiment_score ON stock_sentiment (sentiment_score);

CREATE TABLE IF NOT EXISTS stock_prices (
    ticker TEXT PRIMARY KEY,
    open_price REAL,
    close_price REAL,
    high_price REAL,
    low_price REAL,
    volume INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    predicted_sentiment TEXT,
    predicted_price REAL,
    actual_price REAL,
    prediction_accuracy REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    action TEXT,
    price REAL,
    budget REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio (
    ticker TEXT PRIMARY KEY,
    quantity INTEGER,
    average_buy_price REAL
);

CREATE TABLE IF NOT EXISTS forecast_vs_actual (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    forecasted_price REAL,
    actual_price REAL,
    absolute_deviation REAL,
    percentage_deviation REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS budget (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    available_funds REAL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')
conn.commit()

# Initialize budget if not already present.
def init_budget(initial_amount=100000.0):
    cursor.execute("SELECT COUNT(*) FROM budget;")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO budget (available_funds) VALUES (?);", (initial_amount,))
        conn.commit()

init_budget()

def get_budget():
    cursor.execute("SELECT available_funds FROM budget ORDER BY updated_at DESC LIMIT 1;")
    row = cursor.fetchone()
    return row[0] if row else 0

def update_budget(new_amount):
    cursor.execute("INSERT INTO budget (available_funds) VALUES (?);", (new_amount,))
    conn.commit()

# ---------------------------
# Ticker Validation & JSON Dictionary Functions
# ---------------------------
def similar(a, b):
    """Compute the similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def load_json_dict(json_file="ticker_cache.json"):
    """Load the JSON file containing candidate words and their tickers.
       If the file does not exist, return an empty dictionary.
    """
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logging.error(f"Error loading {json_file}: {e}")
            return {}
    return {}

def save_json_dict(data, json_file="ticker_cache.json"):
    """Save the provided dictionary back to the JSON file."""
    try:
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving to {json_file}: {e}")

def lookup_json_ticker(input_str, json_file="ticker_cache.json"):
    """
    Look up the candidate word or phrase in the JSON dictionary.
    If found and the mapped value is not "notfound", return the ticker.
    If the mapping is "notfound", return None.
    """
    input_str = input_str.strip()
    data = load_json_dict(json_file)
    # Try an exact match.
    if input_str in data:
        ticker = data[input_str]
        if ticker.lower() != "notfound":
            return ticker
        else:
            return None
    return None

def update_json_with_candidate(input_str, ticker, json_file="ticker_cache.json"):
    """
    Update the JSON dictionary by storing the candidate word.
    If a valid ticker was found, store that ticker; otherwise, store "notfound".
    """
    input_str = input_str.strip()
    data = load_json_dict(json_file)
    if input_str not in data:
        data[input_str] = ticker if ticker else "notfound"
        save_json_dict(data, json_file)

def get_ticker_from_csv(input_str, csv_folder="C:/VScode/disser/ticker data"):
    """
    Searches through all CSV files in the specified folder for a ticker based on the input.
    CSV files must have columns: 'Ticker' and 'Company'.
    """
    input_str = input_str.strip()
    csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.lower().endswith(".csv")]
    
    # Direct ticker check.
    if input_str.isupper() and 1 < len(input_str) <= 5:
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if 'Ticker' in df.columns and 'Company' in df.columns:
                    tickers = df['Ticker'].astype(str).str.upper().tolist()
                    if input_str.upper() in tickers:
                        return input_str.upper()
            except Exception as e:
                logging.error(f"Error reading CSV file {file}: {e}")
    
    # Fuzzy matching on company names.
    best_match = (None, 0.0)
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'Ticker' in df.columns and 'Company' in df.columns:
                for _, row in df.iterrows():
                    company_name = str(row['Company']).strip()
                    ticker = str(row['Ticker']).upper().strip()
                    if input_str.lower() in company_name.lower():
                        ratio = similar(input_str.lower(), company_name.lower())
                        if ratio > best_match[1]:
                            best_match = (ticker, ratio)
        except Exception as e:
            logging.error(f"Error reading CSV file {file}: {e}")
    if best_match[0] and best_match[1] > 0.4:
        return best_match[0]
    return None

def validate_ticker(input_str, json_file="ticker_cache.json", csv_folder="C:/VScode/disser/ticker data"):
    """
    Validate the input (ticker or company name) by:
      1. Checking the JSON file for a prior lookup.
         - If found and valid, return the ticker.
         - If found as "notfound", return None.
      2. If not present in JSON, check CSV files.
      3. Finally, use yfinance as a failswitch.
    Update the JSON file with the candidate word and its result.
    """
    candidate = input_str.strip()
    # Step 1: JSON lookup.
    ticker_found = lookup_json_ticker(candidate, json_file)
    if ticker_found:
        try:
            stock = yf.Ticker(ticker_found)
            if stock.info and "regularMarketPrice" in stock.info:
                return ticker_found
        except Exception as e:
            logging.error(f"yfinance validation failed for ticker {ticker_found}: {e}")
            return ticker_found
    # Step 2: Check CSV files.
    ticker_found = get_ticker_from_csv(candidate, csv_folder)
    if ticker_found:
        try:
            stock = yf.Ticker(ticker_found)
            if stock.info and "regularMarketPrice" in stock.info:
                update_json_with_candidate(candidate, ticker_found, json_file)
                return ticker_found
        except Exception as e:
            logging.error(f"yfinance validation failed for ticker {ticker_found}: {e}")
            update_json_with_candidate(candidate, ticker_found, json_file)
            return ticker_found
    # Step 3: Fallback using yfinance directly.
    try:
        stock = yf.Ticker(candidate)
        if stock.info and "regularMarketPrice" in stock.info:
            update_json_with_candidate(candidate, candidate.upper(), json_file)
            return candidate.upper()
    except Exception as e:
        logging.error(f"yfinance failswitch validation failed for input {candidate}: {e}")
    
    update_json_with_candidate(candidate, None, json_file)
    return None

# ---------------------------
# PPO Model & Enhancements
# ---------------------------
class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = 1  
action_size = 3  
policy_net = PPO(state_size, action_size)
target_net = PPO(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
memory = deque(maxlen=2000)

def ppo_decision(sentiment_score):
    normalized_score = (sentiment_score + 1) / 2
    state = torch.tensor([[normalized_score]], dtype=torch.float32)
    with torch.no_grad():
        action_values = policy_net(state)
    action = torch.argmax(action_values).item()
    return ["Buy", "Sell", "Hold"][action]

def compute_advantage(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        try:
            delta = rewards[i] + gamma * values[i+1] - values[i]
        except IndexError:
            delta = rewards[i] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    adv = np.array(advantages)
    if adv.std() != 0:
        adv = (adv - adv.mean()) / adv.std()
    return adv.tolist()

def entropy_loss(probabilities):
    return -torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1))

# ---------------------------
# Sentiment Analysis (Batched)
# ---------------------------
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)['compound']
    sentiment = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
    return sentiment, sentiment_score

def batch_analyze_sentiments(texts):
    return [analyze_sentiment(text) for text in texts]

# ---------------------------
# Stock Price & Ticker Extraction Functions
# ---------------------------
def fetch_stock_prices(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if data.empty:
            logging.error(f"Failed to fetch stock prices for {ticker}")
            return None
        price_data = data.iloc[-1]
        cursor.execute("""
        INSERT INTO stock_prices (ticker, open_price, close_price, high_price, low_price, volume, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            open_price = excluded.open_price,
            close_price = excluded.close_price,
            high_price = excluded.high_price,
            low_price = excluded.low_price,
            volume = excluded.volume,
            timestamp = excluded.timestamp;
        """, (ticker, price_data.Open, price_data.Close, price_data.High, price_data.Low, price_data.Volume, datetime.now()))
        conn.commit()
        return price_data.Close
    except Exception as e:
        logging.error(f"Exception in fetch_stock_prices for {ticker}: {e}")
        return None

def extract_tickers(text):
    """
    Extract candidate words from the text. For each candidate,
    first check the JSON file, then fallback to additional validations.
    """
    potential_candidates = re.findall(r'\b[A-Z][A-Za-z]+\b', text)
    valid_tickers = []
    for candidate in potential_candidates:
        ticker = validate_ticker(candidate)
        if ticker is not None and ticker not in valid_tickers:
            valid_tickers.append(ticker)
    return valid_tickers

# ---------------------------
# Batch Insertion for Reddit Posts
# ---------------------------
def fetch_reddit_posts(subreddit_name, limit=50):
    rows_to_insert = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            post_content = post.title + " " + post.selftext
            sentiment, score = analyze_sentiment(post_content)
            tickers = extract_tickers(post_content)
            for ticker in tickers:
                rows_to_insert.append((ticker, score, sentiment, post_content, subreddit_name, datetime.now()))
            
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                comment_text = comment.body
                sentiment, score = analyze_sentiment(comment_text)
                tickers = extract_tickers(comment_text)
                for ticker in tickers:
                    rows_to_insert.append((ticker, score, sentiment, comment_text, subreddit_name, datetime.now()))
    except Exception as e:
        logging.error(f"Error fetching posts from {subreddit_name}: {e}")
    
    if rows_to_insert:
        try:
            cursor.executemany("""
            INSERT INTO stock_sentiment (ticker, sentiment_score, sentiment, post_content, subreddit, timestamp)
            VALUES (?, ?, ?, ?, ?, ?);
            """, rows_to_insert)
            conn.commit()
        except Exception as e:
            logging.error(f"Batch insertion error: {e}")

# ---------------------------
# LSTM Model for Stock Price Prediction
# ---------------------------
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

lstm_model = LSTMStockPredictor()
lstm_criterion = nn.MSELoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
scaler = MinMaxScaler()

def train_lstm_model(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="6mo")
        if data.empty:
            logging.error(f"No historical data for {ticker}")
            return None
        
        prices = data['Close'].values.reshape(-1, 1)
        prices = scaler.fit_transform(prices)
        X, y = [], []
        seq_length = 10
        for i in range(len(prices) - seq_length):
            X.append(prices[i:i+seq_length])
            y.append(prices[i+seq_length])
        
        X_train = torch.tensor(np.array(X), dtype=torch.float32)
        y_train = torch.tensor(np.array(y), dtype=torch.float32)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for epoch in range(20):
            for batch_X, batch_y in dataloader:
                lstm_optimizer.zero_grad()
                outputs = lstm_model(batch_X)
                loss = lstm_criterion(outputs, batch_y)
                loss.backward()
                lstm_optimizer.step()
        return lstm_model
    except Exception as e:
        logging.error(f"Error training LSTM for {ticker}: {e}")
        return None

def predict_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="6mo")
        if data.empty:
            logging.error(f"No historical data for {ticker}")
            return None
        last_seq = scaler.transform(data['Close'].values[-10:].reshape(-1, 1))
        last_seq = torch.tensor(np.array([last_seq]), dtype=torch.float32)
        with torch.no_grad():
            prediction = lstm_model(last_seq).item()
        return scaler.inverse_transform([[prediction]])[0][0]
    except Exception as e:
        logging.error(f"Error predicting stock price for {ticker}: {e}")
        return None

def log_forecast_vs_actual(ticker):
    forecasted_price = predict_stock_price(ticker)
    actual_price = fetch_stock_prices(ticker)
    if forecasted_price is None or actual_price is None:
        return
    absolute_deviation = abs(forecasted_price - actual_price)
    percentage_deviation = (absolute_deviation / actual_price) * 100
    cursor.execute("""
    INSERT INTO forecast_vs_actual (ticker, forecasted_price, actual_price, absolute_deviation, percentage_deviation, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker, forecasted_price, actual_price, absolute_deviation, percentage_deviation, datetime.now()))
    conn.commit()

# ---------------------------
# Trade Execution & Budget Management
# ---------------------------
def execute_trade(ticker, sentiment_score):
    decision = ppo_decision(sentiment_score)
    current_price = fetch_stock_prices(ticker)
    available_budget = get_budget()
    trade_amount = 1000  # Example fixed trade amount
    
    if current_price is None:
        logging.error(f"Trade aborted: No price for {ticker}")
        return
    
    if decision == "Buy":
        if available_budget >= trade_amount:
            new_budget = available_budget - trade_amount
            update_budget(new_budget)
            cursor.execute("""
            INSERT INTO stock_trades (ticker, action, price, budget, timestamp)
            VALUES (?, 'Buy', ?, ?, ?)
            """, (ticker, current_price, new_budget, datetime.now()))
            conn.commit()
            logging.info(f"Bought {ticker} at {current_price}. New budget: {new_budget}")
        else:
            logging.warning("Insufficient funds to buy.")
    elif decision == "Sell":
        new_budget = available_budget + trade_amount
        update_budget(new_budget)
        cursor.execute("""
        INSERT INTO stock_trades (ticker, action, price, budget, timestamp)
        VALUES (?, 'Sell', ?, ?, ?)
        """, (ticker, current_price, new_budget, datetime.now()))
        conn.commit()
        logging.info(f"Sold {ticker} at {current_price}. New budget: {new_budget}")
    else:
        logging.info(f"Holding {ticker} (No trade executed).")
    return decision

# ---------------------------
# Explainability for PPO Decisions
# ---------------------------
def ppo_predict_np(inputs):
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        outputs = policy_net(inputs_tensor)
    return outputs.numpy()

def explain_ppo_decision(sentiment_score):
    normalized_score = (sentiment_score + 1) / 2
    instance = np.array([[normalized_score]])
    
    # SHAP Explanation
    background = np.array([[0.5]])
    shap_explainer = shap.KernelExplainer(ppo_predict_np, background)
    shap_values = shap_explainer.shap_values(instance)
    print("SHAP values for normalized input", instance[0][0], ":", shap_values)
    
    # Determine which action is chosen by PPO for this instance.
    pred = ppo_predict_np(instance)  # Shape: (1, 3)
    decision_index = np.argmax(pred)
    
    # LIME Explanation: Instead of flattening the output,
    # return the prediction corresponding only to the chosen action.
    lime_background = np.linspace(0, 1, 10).reshape(-1, 1)
    lime_explainer = lime_tabular.LimeTabularExplainer(
        lime_background,
        feature_names=['normalized_score'],
        mode='regression'
    )
    lime_exp = lime_explainer.explain_instance(
        instance[0],
        lambda x: ppo_predict_np(x)[:, decision_index],
        num_features=1
    )
    print("LIME explanation for PPO decision:")
    for feature, weight in lime_exp.as_list():
        print(f"{feature}: {weight}")

# ---------------------------
# Live Dashboard with Streamlit
# ---------------------------
def run_dashboard():
    st.title("AI Trading Bot Dashboard")
    st.subheader("Budget & Recent Trades")
    
    budget_amount = get_budget()
    st.write(f"**Available Budget:** ${budget_amount:.2f}")
    
    trades_df = pd.read_sql_query("SELECT * FROM stock_trades ORDER BY timestamp DESC LIMIT 10;", conn)
    st.dataframe(trades_df)
    
    sentiment_df = pd.read_sql_query("SELECT sentiment, COUNT(*) as count FROM stock_sentiment GROUP BY sentiment;", conn)
    st.bar_chart(sentiment_df.set_index("sentiment"))
    
    st.subheader("Forecast vs Actual Prices")
    forecast_df = pd.read_sql_query("SELECT * FROM forecast_vs_actual ORDER BY timestamp DESC LIMIT 10;", conn)
    st.dataframe(forecast_df)

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Fetch Reddit posts (batch insertion)
    for subreddit in ["stocks", "wallstreetbets", "investing", "IndianStockMarket", "IndiaInvestments", "DalalStreetTalks"]:
        fetch_reddit_posts(subreddit, limit=50)
    
    # Train LSTM and log forecast vs actual for sample tickers
    for ticker in ["AAPL", "TSLA", "GOOGL", "AMZN"]:
        train_lstm_model(ticker)
        log_forecast_vs_actual(ticker)
    
    # Example usage: Analyze sentiment, decide trade, and explain decision.
    sample_text = "This stock is going to skyrocket with amazing growth potential!"
    sentiment, score = analyze_sentiment(sample_text)
    decision = ppo_decision(score)
    logging.info(f"PPO decision for sentiment score {score}: {decision}")
    print(f"PPO decision for sentiment score {score}: {decision}")
    
    explain_ppo_decision(score)
    execute_trade("AAPL", score)  # Example trade execution
    
    run_dashboard()
    
    print("Analysis, forecasting, trading, and explainability completed.")
