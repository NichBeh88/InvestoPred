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
CACHE_EXPIRY = timedelta(minutes=30)
FETCH_LOCK_TIMEOUT = timedelta(minutes=5)

# 🔹 Load stock symbols
@st.cache_data
def load_stock_symbols():
    try:
        sp500_df = pd.read_csv(SP500_CSV_PATH)
        ftse_df = pd.read_csv(FTSE100_CSV_PATH)
        ftse_df["Symbol"] = ftse_df["Symbol"] + ".L"
        return {
            "SP500": sp500_df["Symbol"].tolist(),
            "FTSE100": ftse_df["Symbol"].tolist()
        }
    except Exception as e:
        st.error(f"⚠️ Error loading stock symbols: {e}")
        return {"SP500": [], "FTSE100": []}

# 🔒 Lock helpers
def is_fetching_in_progress(index_name):
    ref = db.collection("stock_cache").document(f"{index_name}_fetch_status")
    snap = ref.get()
    if snap.exists:
        start = datetime.fromisoformat(snap.to_dict()["StartTime"])
        return datetime.now(timezone.utc) - start < FETCH_LOCK_TIMEOUT
    return False

def update_fetch_status(index_name, status):
    ref = db.collection("stock_cache").document(f"{index_name}_fetch_status")
    if status == "start":
        ref.set({"StartTime": datetime.now(timezone.utc).isoformat()})
    elif status == "done":
        ref.delete()

# 🔁 Save to Firestore
def save_to_firestore(index_name, stock_data):
    doc_ref = db.collection("stock_cache").document(index_name)
    doc_ref.set({
        "stocks": stock_data.to_dict(orient="records"),
        "updated_at": datetime.now(timezone.utc).isoformat()
    })

# 🚀 Fetch using yf.download
def fetch_and_store_stock_data(index_name):
    if is_fetching_in_progress(index_name):
        return

    update_fetch_status(index_name, "start")
    symbols = load_stock_symbols().get(index_name, [])
    stock_data = []

    try:
        # Fetch batch price data for all symbols
        df = yf.download(
            tickers=symbols,
            period="1d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )

        for symbol in symbols:
            try:
                price = df[symbol]["Close"][-1] if symbol in df else None
                stock_data.append({
                    "Symbol": symbol,
                    "Price": round(price, 2) if price else None,
                    "Company Name": symbol,
                    "Sector": "Unknown",
                    "P/E Ratio": None,
                    "Dividend Yield (%)": None,
                    "EPS": None,
                    "Market Cap": None
                })
            except Exception as e:
                print(f"⚠️ Error with {symbol}: {e}")

        if stock_data:
            save_to_firestore(index_name, pd.DataFrame(stock_data))
            print(f"✅ {index_name} updated.")

    except Exception as e:
        print(f"❌ Fetch failed for {index_name}: {e}")
    finally:
        update_fetch_status(index_name, "done")

# 🧠 Main function for external use
def get_cached_stock_data(index_name):
    doc_ref = db.collection("stock_cache").document(index_name)
    doc = doc_ref.get()

    if doc.exists:
        data = doc.to_dict()
        if "updated_at" not in data:
            return pd.DataFrame(data.get("stocks", []))

        last_updated = datetime.fromisoformat(data["updated_at"])
        if datetime.now(timezone.utc) - last_updated < CACHE_EXPIRY:
            return pd.DataFrame(data["stocks"])

        if not is_fetching_in_progress(index_name):
            print(f"⏳ Refreshing {index_name} cache...")
            threading.Thread(target=fetch_and_store_stock_data, args=(index_name,), daemon=True).start()

        return pd.DataFrame(data["stocks"])

    if not is_fetching_in_progress(index_name):
        print(f"📡 No cache for {index_name}. Fetching...")
        threading.Thread(target=fetch_and_store_stock_data, args=(index_name,), daemon=True).start()

    return pd.DataFrame()  # Placeholder


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
