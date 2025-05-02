import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import time
import firebase_admin
from firebase_admin import credentials, firestore
import threading
import requests

# 🔹 File paths for stock symbol CSVs
SP500_CSV_PATH = "sp500_companies.csv"
FTSE100_CSV_PATH = "FTSE100_Constituents.csv"

# 🔹 Firebase Initialization
if not firebase_admin._apps:
    private_key = st.secrets["FIREBASE"]["private_key"]
    client_email = st.secrets["FIREBASE"]["client_email"]
    project_id = st.secrets["FIREBASE"]["project_id"]

    service_account_info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key": private_key,
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40investopred.iam.gserviceaccount.com"
    }

    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()
CACHE_EXPIRY = timedelta(minutes=60)
FETCH_LOCK_TIMEOUT = timedelta(minutes=5)

# 🔹 Load stock symbols
def load_stock_symbols():
    try:
        sp500_df = pd.read_csv(SP500_CSV_PATH)
        sp500_tickers = sp500_df["Symbol"].tolist()

        ftse100_df = pd.read_csv(FTSE100_CSV_PATH)
        ftse100_df["Symbol"] = ftse100_df["Symbol"] + ".L"  # Append ".L" for Yahoo Finance
        ftse100_tickers = ftse100_df["Symbol"].tolist()

        return {"SP500": sp500_tickers, "FTSE100": ftse100_tickers}
    except Exception as e:
        st.error(f"⚠️ Error loading stock symbols: {e}")
        return {"SP500": [], "FTSE100": []}

# Function to check if a fetch is already in progress
def is_fetching_in_progress(index_name):
    fetch_status_ref = db.collection("stock_cache").document(f"{index_name}_fetch_status")
    fetch_status = fetch_status_ref.get()
    if fetch_status.exists:
        data = fetch_status.to_dict()
        last_fetch_start = datetime.fromisoformat(data["StartTime"])
        return datetime.now(timezone.utc) - last_fetch_start < FETCH_LOCK_TIMEOUT
    return False


# Function to update Firestore fetch status
def update_fetch_status(index_name, status):
    fetch_status_ref = db.collection("stock_cache").document(f"{index_name}_fetch_status")
    if status == "start":
        fetch_status_ref.set({"StartTime": datetime.now(timezone.utc).isoformat()})
    elif status == "done":
        fetch_status_ref.delete()

# 🔹 Function to fetch cached stock data from Firestore
def get_firestore_cache(index_name):
    doc_ref = db.collection("stock_cache").document(index_name)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        last_updated = datetime.strptime(data["LastUpdated"], "%Y-%m-%d %H:%M:%S")

        if datetime.now() - last_updated < CACHE_EXPIRY:
            return pd.DataFrame(data["StockData"])  # Return cached data

    return None  # Cache expired or missing

# 🔹 Function to save stock data to Firestore
def save_to_firestore(index_name, stock_data):
    doc_ref = db.collection("stock_cache").document(index_name)
    doc_ref.set({
        "StockData": stock_data.to_dict(orient="records"),
        "LastUpdated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    print(f"✅ Firestore cache updated for {index_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def fetch_and_store_stock_data(index_name):
    if is_fetching_in_progress(index_name):
        return

    update_fetch_status(index_name, "start")
    stock_symbols = load_stock_symbols().get(index_name, [])
    stock_data = []

    batch_size = 25  # smaller to reduce pressure

    try:
        for i in range(0, len(stock_symbols), batch_size):
            batch_tickers = stock_symbols[i:i + batch_size]
            tickers = yf.Tickers(" ".join(batch_tickers))

            for symbol in batch_tickers:
                try:
                    ticker = tickers.tickers[symbol]

                    # Use fast_info for speed & safety
                    fast_info = getattr(ticker, "fast_info", {})
                    info = ticker.info  # fallback

                    stock_data.append({
                        "Symbol": symbol,
                        "Company Name": info.get("shortName", "N/A"),
                        "Sector": info.get("sector", "Unknown"),
                        "Price": fast_info.get("last_price", info.get("previousClose", None)),
                        "P/E Ratio": info.get("trailingPE", None),
                        "Dividend Yield (%)": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
                        "EPS": info.get("trailingEps", None),
                        "Market Cap": info.get("marketCap", None),
                    })

                    time.sleep(0.2)  # small delay between calls

                except Exception as e:
                    print(f"⚠️ Error fetching {symbol}: {e}")
                    continue

        if stock_data:
            db.collection("stock_cache").document(index_name).set({
                "stocks": stock_data,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"✅ {index_name} stock data updated in Firestore.")

    except Exception as e:
        print(f"🔥 Error during batch fetch for {index_name}: {e}")

    finally:
        update_fetch_status(index_name, "done")


def get_cached_stock_data(index_name):
    doc_ref = db.collection("stock_cache").document(index_name)
    doc = doc_ref.get()

    if doc.exists:
        cache_data = doc.to_dict()
        if "updated_at" not in cache_data:
            print(f"⚠️ Missing 'updated_at' in {index_name} cache.")
            return pd.DataFrame(cache_data.get("stocks", []))

        last_updated = datetime.fromisoformat(cache_data["updated_at"])

        if datetime.now(timezone.utc) - last_updated < CACHE_EXPIRY:
            return pd.DataFrame(cache_data["stocks"])

        if is_fetching_in_progress(index_name):
            print(f"🔁 Fetch in progress for {index_name}, using stale cache.")
            return pd.DataFrame(cache_data["stocks"])

        print(f"🔄 Cache expired for {index_name}, fetching in background...")
        threading.Thread(target=fetch_and_store_stock_data, args=(index_name,), daemon=True).start()
        return pd.DataFrame(cache_data["stocks"])

    else:
        if is_fetching_in_progress(index_name):
            print(f"🕒 No cache yet, fetch already started for {index_name}.")
            return pd.DataFrame()

        print(f"🚀 Starting fetch for new cache: {index_name}")
        threading.Thread(target=fetch_and_store_stock_data, args=(index_name,), daemon=True).start()
        return pd.DataFrame()


# Top gainers and losers cache

# 🔹 FMP API keys
API_KEY = st.secrets["FIREBASE"]["fmp_api_key"]
CACHE_DURATION_HOURS = 12  # Set cache expiry time to 12 hours

# 🔹 Function to fetch and cache data for Top Gainers or Top Losers
def fetch_and_cache_fmp_data(endpoint):
    doc_ref = db.collection("market").document(endpoint)
    doc = doc_ref.get()

    now = datetime.utcnow()

    # Check if cached data exists and if it is still fresh
    if doc.exists:
        data = doc.to_dict()
        timestamp = datetime.fromisoformat(data["timestamp"])
        if now - timestamp < timedelta(hours=CACHE_DURATION_HOURS):
            return data["results"]  # Return cached data if within the expiry time

    # Fetch new data from the API if cache is expired or missing
    url = f"https://financialmodelingprep.com/api/v3/stock_market/{endpoint}?apikey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json()
        doc_ref.set({
            "timestamp": now.isoformat(),
            "results": results
        })
        return results
    else:
        # Optional fallback: return old cache data if API fails
        return data["results"] if doc.exists else []

def get_top_gainers():
    return fetch_and_cache_fmp_data("gainers")

def get_top_losers():
    return fetch_and_cache_fmp_data("losers")
