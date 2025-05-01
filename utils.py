import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import time
import firebase_admin
from firebase_admin import credentials, firestore
import threading

# 🔹 File paths for stock symbol CSVs
SP500_CSV_PATH = "/Users/nicholasbeh/Downloads/Stock_Web/sp500_companies.csv"
FTSE100_CSV_PATH = "/Users/nicholasbeh/Downloads/Stock_Web/FTSE100_Constituents.csv"

# 🔹 Initialize Firebase Firestore
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/nicholasbeh/Downloads/Stock_Web/investopred-firebase-adminsdk-fbsvc-6b373aa545.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
CACHE_EXPIRY = timedelta(minutes=30)  # Cache expiry time
FETCH_LOCK_TIMEOUT = timedelta(minutes=5)  # Prevents multiple fetches stacking up

# 🔹 Function to load stock symbols from CSV
@st.cache_data
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
        
        # If fetch started within timeout period, return True (fetch in progress)
        if datetime.now(timezone.utc) - last_fetch_start < FETCH_LOCK_TIMEOUT:
            return True
        
    return False

# Function to update Firestore fetch status
def update_fetch_status(index_name, status):
    fetch_status_ref = db.collection("stock_cache").document(f"{index_name}_fetch_status")
    
    if status == "start":
        fetch_status_ref.set({"StartTime": datetime.now(timezone.utc).isoformat()})
    elif status == "done":
        fetch_status_ref.delete()  # Remove fetch lock after completion

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
    """Fetches fresh stock data & stores in Firestore."""
    if is_fetching_in_progress(index_name):
        return  # 🚫 Skip fetch if another process is already fetching
    
    update_fetch_status(index_name, "start")  # ✅ Mark fetch as started    

    stock_symbols = load_stock_symbols().get(index_name, [])
    
    batch_size = 50  # ✅ Fetch in small batches to prevent rate limit
    stock_data = []
    
    try:
        for i in range(0, len(stock_symbols), batch_size):
            batch_tickers = stock_symbols[i:i+batch_size]
            for symbol in batch_tickers:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    stock_data.append({
                        "Symbol": symbol,
                        "Company Name": info.get("shortName", "N/A"),
                        "Sector": info.get("sector", "Unknown"),
                        "Price": info.get("previousClose", None),
                        "P/E Ratio": info.get("trailingPE", None),
                        "Dividend Yield (%)": info.get("dividendYield", 0),
                        "EPS": info.get("trailingEps", None),
                        "Market Cap": info.get("marketCap", None),
                    })
                    time.sleep(1)  # ✅ Small delay to prevent rate limits
                except Exception as e:
                    print(f"⚠️ Error fetching {symbol}: {e}")

        if stock_data:
            db.collection("stock_cache").document(index_name).set({
                "stocks": stock_data,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"✅ {index_name} stock data updated in Firestore.")

    except Exception as e:
        print(f"⚠️ Error fetching {index_name}: {e}")
    
    finally:
        update_fetch_status(index_name, "done")  # ✅ Remove fetch lock after completion


def get_cached_stock_data(index_name):
    """Loads stock data from Firestore; fetches new data if expired (runs in background)."""
    doc_ref = db.collection("stock_cache").document(index_name)
    doc = doc_ref.get()

    if doc.exists:
        cache_data = doc.to_dict()
        # ✅ Check if 'updated_at' exists before using it
        if "updated_at" not in cache_data:
            print(f"⚠️ Missing 'updated_at' in {index_name} cache. Returning last known data.")
            return pd.DataFrame(cache_data.get("stocks", []))  # Return whatever is available

        last_updated = datetime.fromisoformat(cache_data["updated_at"])

        # ✅ If data is fresh, return it
        if datetime.now(timezone.utc) - last_updated < CACHE_EXPIRY:
            return pd.DataFrame(cache_data["stocks"])
        
        # ✅ If fetch is already in progress, return old cache instead of stacking requests
        if is_fetching_in_progress(index_name):
            print(f"⚠️ Fetch already in progress for {index_name}. Returning existing cache.")
            return pd.DataFrame(cache_data["stocks"])
        

        # ✅ Cache expired → Fetch fresh data in background (but return old cache)
        print(f"🔄 {index_name} cache expired, fetching new data in background...")
        threading.Thread(target=fetch_and_store_stock_data, args=(index_name,)).start()
        return pd.DataFrame(cache_data["stocks"])
    
    # ✅ If no cached data exists, fetch immediately
    print(f"⚠️ No cache found for {index_name}, fetching fresh data...")
    fetch_and_store_stock_data(index_name)
    return get_cached_stock_data(index_name)  # Load new cache after fetch
