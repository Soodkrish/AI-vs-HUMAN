import sys
import sqlite3
import requests
from bs4 import BeautifulSoup
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import spacy
from transformers import pipeline
from tf_keras import Sequential
from tf_keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import re
import json
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import backtrader as bt
import subprocess

# ===============================================
# Local Ticker Directory: Load from Multiple CSV Files
# ===============================================
def load_local_tickers_from_folder(folder_path):
    """
    Look for all CSV files in folder_path and combine them.
    Each CSV is expected to have columns 'Company' and 'Ticker'.
    Returns a dictionary mapping lowercase company names to ticker symbols.
    """
    ticker_dict = {}
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' not found. No local tickers loaded.")
        return ticker_dict
    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                for idx, row in df.iterrows():
                    company = str(row["Company"]).strip().lower()
                    ticker = str(row["Ticker"]).strip()
                    ticker_dict[company] = ticker
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return ticker_dict

# Set the folder path containing CSV files (update as needed)
LOCAL_TICKER_FOLDER = "C:/VScode/disser/ticker data"
LOCAL_TICKERS = load_local_tickers_from_folder(LOCAL_TICKER_FOLDER)

# ===============================================
# Fuzzy Matching Helper
# ===============================================
def fuzzy_match_ticker(candidate, local_dict, threshold=0.5):
    """
    Returns a ticker if the candidate is a substring of a key (or vice versa)
    and its length is at least threshold*len(key). This is a simple fuzzy matching function.
    """
    candidate = candidate.lower()
    for company, ticker in local_dict.items():
        if len(candidate) >= threshold * len(company) and (candidate in company or company in candidate):
            return ticker
    return None

# ===============================================
# Caching Functions for Ticker Extraction
# ===============================================
CACHE_FILE = "ticker_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print("Error loading cache:", e)
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print("Error saving cache:", e)

def search_for_ticker(word):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={word}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        if "quotes" in data:
            for quote in data["quotes"]:
                if quote.get("symbol") and quote.get("shortname"):
                    if word.lower() in quote.get("shortname").lower():
                        return quote["symbol"].upper()
        return None
    except Exception as e:
        print(f"Error searching ticker for '{word}': {e}")
        return None

def search_for_ticker_with_cache(word):
    cache = load_cache()
    if word in cache:
        result = cache[word]
        return None if result == "not found" else result
    ticker = search_for_ticker(word)
    if ticker is None:
        cache[word] = "not found"
    else:
        cache[word] = ticker
    save_cache(cache)
    return ticker

# ===============================================
# Ticker Lookup: Local Directory (with Fuzzy Matching), then Fallback
# ===============================================
STOCK_LIST_URL = "https://stockanalysis.com/stocks/"
def get_stock_list():
    if LOCAL_TICKERS:
        return LOCAL_TICKERS
    try:
        response = requests.get(STOCK_LIST_URL, headers=HEADERS)
        if response.status_code != 200:
            print("Failed to fetch stock list online")
            return {}
        soup = BeautifulSoup(response.text, "html.parser")
        stocks = {}
        for row in soup.select("table tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                ticker = cols[0].text.strip()
                name = cols[1].text.strip()
                stocks[name.lower()] = ticker
        return stocks
    except Exception as e:
        print(f"Error fetching online stock list: {e}")
        return {}

STOCKS = get_stock_list()

def extract_ticker(text):
    """
    Extract ticker by:
      1. Using Spacy NER to extract candidate company names.
      2. Checking against local STOCKS (exact and fuzzy matching).
      3. Falling back to online search using cache.
    """
    doc = nlp(text)
    companies = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]
    for company in companies:
        company_lower = company.lower()
        if company_lower in STOCKS:
            return STOCKS[company_lower]
        fm_ticker = fuzzy_match_ticker(company_lower, STOCKS)
        if fm_ticker:
            return fm_ticker
        ticker = search_for_ticker_with_cache(company)
        if ticker:
            return ticker
    words = text.split()
    for word in words:
        word_clean = re.sub(r'[^\w\s]', '', word).lower()
        if not word_clean:
            continue
        if word_clean in STOCKS:
            return STOCKS[word_clean]
        fm_ticker = fuzzy_match_ticker(word_clean, STOCKS)
        if fm_ticker:
            return fm_ticker
        ticker = search_for_ticker_with_cache(word_clean)
        if ticker:
            return ticker
    return None

# ===============================================
# Global Setup
# ===============================================
nlp = spacy.load("en_core_web_sm")
DB_PATH = "news_data_new_v9.db"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/123.0.0.0 Safari/537.36")
}
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert", batch_size=8)

# ===============================================
# Database & News Scraping Functions
# ===============================================
def connect_db():
    return sqlite3.connect(DB_PATH)

def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS news (
                        id INTEGER PRIMARY KEY,
                        source TEXT,
                        title TEXT UNIQUE,
                        url TEXT,
                        published TEXT,
                        sentiment TEXT,
                        ticker TEXT
                      )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY,
                        ticker TEXT,
                        predicted_price REAL,
                        trade_decision TEXT,
                        final_net_worth REAL,
                        volatility REAL,
                        trade_timestamp TEXT
                      )''')
    conn.commit()
    conn.close()
    print("Database and tables created/updated successfully.")

def fetch_news(url, source):
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        articles = []
        for item in soup.find_all("item"):
            title = item.title.text
            link = item.link.text
            pub_date = item.pubDate.text
            articles.append((source, title, link, pub_date, extract_ticker(title)))
        return articles
    except Exception as e:
        print(f"Error fetching {source}: {e}")
        return []

def fetch_google_news():
    return fetch_news("https://news.google.com/rss/search?q=stock+market", "Google News")

def fetch_bloomberg_news():
    return fetch_news("https://news.google.com/rss/search?q=Bloomberg+stock+market", "Bloomberg")

def fetch_cnbc_news():
    return fetch_news("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC")

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]["label"]

def store_news(articles):
    conn = connect_db()
    cursor = conn.cursor()
    for source, title, url, published, ticker in articles:
        sentiment = analyze_sentiment(title)
        try:
            cursor.execute("INSERT INTO news (source, title, url, published, sentiment, ticker) VALUES (?, ?, ?, ?, ?, ?)", 
                           (source, title, url, published, sentiment, ticker))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()

def analyze_headlines_topic():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM news")
    headlines = [row[0] for row in cursor.fetchall()]
    conn.close()
    topics = []
    for headline in headlines:
        doc = nlp(headline)
        topics.extend([ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]])
    freq = Counter(topics)
    top_topics = freq.most_common(5)
    print("Top discussed topics in headlines:")
    for topic, count in top_topics:
        print(f"{topic}: {count} times")
    return top_topics

# ===============================================
# Memory Retention: Aggregate past sentiment for a ticker
# ===============================================
def get_ticker_news_memory(ticker):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment FROM news WHERE ticker=?", (ticker,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return 1.0
    score_map = {"positive": 1, "negative": -1, "neutral": 0}
    scores = [score_map.get(row[0].lower(), 0) for row in rows]
    avg_score = np.mean(scores)
    adjustment = 1 + (avg_score * 0.02)
    return adjustment

# ===============================================
# Stock Price Forecasting (LSTM)
# ===============================================
def get_historical_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        if hist.empty:
            return None
        return hist['Close'].values
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def create_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(50, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X_train, y_train = [], []
    for i in range(50, len(scaled_data)):
        X_train.append(scaled_data[i-50:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = create_lstm_model()
    model.fit(X_train, y_train, batch_size=1, epochs=5)
    return model, scaler

def predict_stock_price(model, scaler, data):
    last_50_days = data[-50:].reshape(-1, 1)
    scaled_data = scaler.transform(last_50_days)
    X_test = np.reshape(scaled_data, (1, 50, 1))
    predicted_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_price)[0, 0]

# ===============================================
# Reinforcement Learning with DQN for Trading
# ===============================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

MODEL_FILE = "dqn_model.pt"

def load_dqn_model(state_size, action_size):
    if os.path.exists(MODEL_FILE):
        print("Loading persistent DQN model...")
        with torch.serialization.safe_globals({"__main__.DQN": DQN}):
            model = torch.load(MODEL_FILE, weights_only=False)
        model.eval()
        return model
    else:
        print("No persistent DQN model found. Creating a new one...")
        return DQN(state_size, action_size)

class TradingEnv(gym.Env):
    def __init__(self, historical_prices, max_shares=20, stop_loss=0.95, take_profit=1.10):
        super(TradingEnv, self).__init__()
        self.historical_prices = historical_prices
        self.current_step = 50
        self.balance = 10000
        self.position = 0
        self.max_shares = max_shares
        self.initial_balance = 10000
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
        self.net_worth_history = [self.initial_balance]
    
    def _get_observation(self):
        return self.historical_prices[self.current_step-50:self.current_step]
    
    def step(self, action):
        current_price = self.historical_prices[self.current_step]
        if action == 1:
            # Dynamic Position Sizing using a simple Kelly Criterion
            expected_return = (self.historical_prices[-1] - current_price) / current_price
            volatility = np.std(np.diff(self.historical_prices) / self.historical_prices[:-1])
            variance = volatility**2 if volatility > 0 else 1e-6
            fraction = expected_return / variance
            fraction = min(max(fraction, 0), 1)
            amount_to_invest = self.balance * fraction
            shares_to_buy = int(amount_to_invest // current_price)
            shares_to_buy = min(shares_to_buy, self.max_shares)
            self.position += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2 and self.position > 0:
            self.balance += self.position * current_price
            self.position = 0
        
        self.current_step += 1
        net_worth = self.balance + self.position * self.historical_prices[self.current_step-1]
        self.net_worth_history.append(net_worth)
        if net_worth <= self.initial_balance * self.stop_loss or net_worth >= self.initial_balance * self.take_profit:
            if self.position > 0:
                self.balance += self.position * current_price
                self.position = 0
                net_worth = self.balance
        done = self.current_step >= len(self.historical_prices)
        reward = net_worth - self.initial_balance
        obs = self._get_observation()
        return obs, reward, done, {}
    
    def reset(self):
        self.current_step = 50
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth_history = [self.initial_balance]
        return self.historical_prices[:50]
    
    def render(self, mode='human'):
        net_worth = self.balance + self.position * self.historical_prices[self.current_step-1]
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.position}, Net Worth: {net_worth}")

def train_dqn(env, episodes=50, initial_epsilon=1.0, min_epsilon=0.1, decay_rate=0.99, batch_size=32, continue_training=True):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = load_dqn_model(state_size, action_size)
    if continue_training:
        optimizer = optim.Adam(dqn.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        replay_buffer = collections.deque(maxlen=1000)
        epsilon = initial_epsilon
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if random.random() < epsilon:
                    action = random.choice([0, 1, 2])
                else:
                    with torch.no_grad():
                        q_values = dqn(state_tensor)
                    action = int(torch.argmax(q_values).item())
                next_state, reward, done, _ = env.step(action)
                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                if len(replay_buffer) > batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.FloatTensor(np.array(states))
                    actions = torch.LongTensor(np.array(actions))
                    rewards = torch.FloatTensor(np.array(rewards))
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(np.array(dones))
                    current_q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
                    next_q = dqn(next_states).max(1)[0].detach()
                    target_q = rewards + (0.99 * next_q * (1 - dones))
                    loss = criterion(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            epsilon = max(min_epsilon, epsilon * decay_rate)
            print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        torch.save(dqn, MODEL_FILE)
    return dqn

def trade_with_dqn(dqn, historical_prices):
    env = TradingEnv(historical_prices, max_shares=20)
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        action = int(torch.argmax(q_values).item())
        state, reward, done, _ = env.step(action)
        env.render()
    final_net_worth = env.balance + env.position * env.historical_prices[env.current_step-1]
    print(f"Final net worth: {final_net_worth}")
    return final_net_worth, env.net_worth_history

# ===============================================
# Database Enhancements for Trade Logging
# ===============================================
def store_trade(ticker, predicted_price, trade_decision, final_net_worth, volatility):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("INSERT INTO trades (ticker, predicted_price, trade_decision, final_net_worth, volatility, trade_timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                   (ticker, float(predicted_price), trade_decision, final_net_worth, volatility, timestamp))
    conn.commit()
    conn.close()

def compute_volatility(data):
    returns = np.diff(data) / data[:-1]
    return np.std(returns)

# ===============================================
# Additional Performance Metrics Functions
# ===============================================
def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_return = np.mean(returns) - risk_free_rate
    std_dev = np.std(returns)
    return excess_return / std_dev if std_dev != 0 else 0

def compute_sortino_ratio(returns, risk_free_rate=0.0):
    downside_returns = returns[returns < risk_free_rate]
    expected_return = np.mean(returns) - risk_free_rate
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    return expected_return / downside_std if downside_std != 0 else 0

def compute_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd

def compute_calmar_ratio(returns, max_drawdown, trading_days=252):
    annual_return = np.mean(returns) * trading_days
    return annual_return / max_drawdown if max_drawdown != 0 else float('inf')

def compute_omega_ratio(returns, target=0):
    positive_sum = np.sum(returns[returns > target] - target)
    negative_sum = np.sum(target - returns[returns < target])
    return positive_sum / negative_sum if negative_sum != 0 else float('inf')

def compute_ulcer_index(portfolio_values):
    peak = portfolio_values[0]
    drawdowns = []
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        drawdowns.append(drawdown)
    return np.sqrt(np.mean(np.square(drawdowns)))

# ===============================================
# Enhanced Data Integration: Live Data & Macroeconomics
# ===============================================
def fetch_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice", None)
        return price
    except Exception as e:
        print(f"Error fetching live price for {ticker}: {e}")
        return None

def get_macro_indicators():
    return {"GDP_growth": 0.02, "inflation": 0.025, "interest_rate": 0.01}

# ===============================================
# New Feature: Fetch Company Financial Articles via Google News
# ===============================================
def fetch_company_financial_articles_google(ticker):
    """
    Performs a simple Google News RSS query for "{ticker} financial".
    It fetches up to 5 articles, analyzes their sentiment, and returns an adjustment multiplier.
    """
    url = f"https://news.google.com/rss/search?q={ticker}+financial"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        articles = []
        for item in soup.find_all("item")[:5]:
            title = item.title.text
            articles.append(title)
        if not articles:
            return 1.0
        sentiments = [analyze_sentiment(article) for article in articles]
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        scores = [score_map.get(s.lower(), 0) for s in sentiments]
        avg_score = np.mean(scores)
        adjustment = 1 + (avg_score * 0.01)  # 1% adjustment per sentiment point
        print(f"Financial articles adjustment for {ticker}: {adjustment}")
        return adjustment
    except Exception as e:
        print(f"Error fetching financial articles for {ticker}: {e}")
        return 1.0

# ===============================================
# New Feature: Dynamic Position Sizing using Kelly Criterion
# ===============================================
def calculate_kelly_shares(balance, current_price, forecast_price, volatility, max_shares=20):
    """
    Uses a simple Kelly criterion to decide a fraction of the balance to invest.
    Returns the number of shares to buy, capped at max_shares.
    """
    expected_return = (forecast_price - current_price) / current_price
    variance = volatility**2 if volatility > 0 else 1e-6
    fraction = expected_return / variance
    fraction = min(max(fraction, 0), 1)
    amount_to_invest = balance * fraction
    shares = int(amount_to_invest // current_price)
    return min(shares, max_shares)

# ===============================================
# New Feature: Live Video Understanding (Simple CPU-Based)
# ===============================================
def analyze_live_video(video_link):
    """
    Downloads audio from the provided video link using yt-dlp,
    transcribes it using the Whisper small model on CPU,
    analyzes sentiment, and returns a confidence multiplier.
    This is kept simple to be CPU-efficient.
    """
    temp_audio = "temp_audio.mp3"
    try:
        print("Downloading audio from video link...")
        subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", temp_audio, video_link], check=True)
    except Exception as e:
        print("Error downloading audio:", e)
        return 1.0

    try:
        import whisper
        print("Transcribing audio with Whisper (small, CPU-only)...")
        model_whisper = whisper.load_model("small", device="cpu")
        result = model_whisper.transcribe(temp_audio)
        transcript = result["text"]
        print("Transcript:", transcript)
        live_sentiment = analyze_sentiment(transcript)
        score_map = {"positive": 1, "negative": -1, "neutral": 0}
        confidence_multiplier = 1 + (score_map.get(live_sentiment.lower(), 0) * 0.01)
        print("Live video sentiment:", live_sentiment)
    except Exception as e:
        print("Error during live video analysis:", e)
        confidence_multiplier = 1.0
    try:
        os.remove(temp_audio)
    except:
        pass
    return confidence_multiplier

# ===============================================
# New Feature: Backtesting Framework Using Backtrader
# ===============================================
class SMACrossStrategy(bt.Strategy):
    params = (("sma_period", 20),)
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)
    def next(self):
        if self.data.close[0] > self.sma[0] and not self.position:
            self.buy()
        elif self.data.close[0] < self.sma[0] and self.position:
            self.sell()

def run_backtest(ticker):
    class SMACrossStrategy(bt.Strategy):
        params = (("sma_period", 20),)
        def __init__(self):
            self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)
        def next(self):
            if self.data.close[0] > self.sma[0] and not self.position:
                self.buy()
            elif self.data.close[0] < self.sma[0] and self.position:
                self.sell()
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SMACrossStrategy)

    from_date = datetime.datetime.now() - datetime.timedelta(days=5*365)
    to_date = datetime.datetime.now()

    # Set auto_adjust=False so that the returned DataFrame has proper column names.
    df = yf.download(ticker, start=from_date, end=to_date, auto_adjust=False)
    if df.empty:
        print(f"No data found for {ticker}")
        return None

    df.dropna(inplace=True)
    
    # Ensure that column names are strings
    df.columns = [str(col) for col in df.columns]

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)
    print(f"Starting Backtest for {ticker} with portfolio value {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value for {ticker}: {final_value:.2f}")
    return final_value



# ===============================================
# Main Pipeline: News, Forecasting, Trading & Reporting
# ===============================================
def trade_stocks():
    analyze_headlines_topic()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker, sentiment FROM news WHERE sentiment='positive'")
    rows = cursor.fetchall()
    tickers = [row[0] for row in rows if row[0] is not None]
    sentiments = {row[0]: row[1] for row in rows if row[0] is not None}
    conn.close()
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        data = get_historical_data(ticker)
        if data is None or len(data) < 50:
            continue
        
        # LSTM Forecasting
        lstm_model, scaler = train_lstm(data)
        predicted_price = predict_stock_price(lstm_model, scaler, data)
        
        # Basic sentiment weighting: +5% for positive, -5% for negative
        sentiment = sentiments.get(ticker, "neutral")
        if sentiment.lower() == "positive":
            adjusted_price = predicted_price * 1.05
        elif sentiment.lower() == "negative":
            adjusted_price = predicted_price * 0.95
        else:
            adjusted_price = predicted_price
        
        # Memory-based adjustment from past articles
        memory_adjustment = get_ticker_news_memory(ticker)
        final_adjusted_price = adjusted_price * memory_adjustment
        
        # Additional adjustment based on financial articles from Google News
        financial_adjustment = fetch_company_financial_articles_google(ticker)
        final_adjusted_price *= financial_adjustment
        
        # Optionally, if a video link is provided (via command-line), incorporate its confidence multiplier
        if len(sys.argv) > 1:
            video_link = sys.argv[1]
            video_multiplier = analyze_live_video(video_link)
            print(f"Live video multiplier for {ticker}: {video_multiplier}")
            final_adjusted_price *= video_multiplier
        
        final_adjusted_price = float(final_adjusted_price)
        print(f"Predicted Price for {ticker}: {final_adjusted_price}")
        
        # Compute volatility
        volatility = compute_volatility(data)
        print(f"Volatility for {ticker}: {volatility}")
        
        # Dynamic position sizing via Kelly Criterion
        current_price = data[-1]
        balance = 10000  # starting balance
        dynamic_shares = calculate_kelly_shares(balance, current_price, final_adjusted_price, volatility, max_shares=20)
        print(f"Dynamic position sizing suggests buying {dynamic_shares} shares.")
        
        # DQN Trading Agent with dynamic share limit
        env = TradingEnv(data, max_shares=dynamic_shares)
        dqn_model = train_dqn(env, episodes=50)
        final_net_worth, net_worth_history = trade_with_dqn(dqn_model, data)
        
        if final_net_worth > 10000:
            trade_decision = "BUY"
        elif final_net_worth < 10000:
            trade_decision = "SELL"
        else:
            trade_decision = "HOLD"
        print(f"Trade Decision for {ticker}: {trade_decision}")
        
        store_trade(ticker, final_adjusted_price, trade_decision, final_net_worth, volatility)
        
        # Performance Metrics
        port_returns = np.diff(net_worth_history) / np.array(net_worth_history[:-1])
        sharpe = compute_sharpe_ratio(port_returns)
        sortino = compute_sortino_ratio(port_returns)
        max_dd = compute_max_drawdown(net_worth_history)
        calmar = compute_calmar_ratio(port_returns, max_dd)
        omega = compute_omega_ratio(port_returns)
        ulcer = compute_ulcer_index(net_worth_history)
        print(f"Sharpe Ratio: {sharpe:.3f}, Sortino Ratio: {sortino:.3f}, Max Drawdown: {max_dd:.3f}")
        print(f"Calmar Ratio: {calmar:.3f}, Omega Ratio: {omega:.3f}, Ulcer Index: {ulcer:.3f}")
        
        live_price = fetch_live_price(ticker)
        macro = get_macro_indicators()
        print(f"Live Price for {ticker}: {live_price}")
        print(f"Macro Indicators: {macro}")
        
        print("Running backtest for strategy evaluation...")
        run_backtest(ticker)

def main():
    create_table()
    all_articles = fetch_google_news() + fetch_bloomberg_news() + fetch_cnbc_news()
    store_news(all_articles)
    trade_stocks()

if __name__ == "__main__":
    main()
