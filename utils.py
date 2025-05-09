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
# 🔹 Load stock symbols
def load_stock_symbols():
    try:
        sp500_df = pd.read_csv(SP500_CSV_PATH)
        sp500_tickers = sp500_df["Symbol"].tolist()

        ftse100_df = pd.read_csv(FTSE100_CSV_PATH)
        ftse100_df["Symbol"] = ftse100_df["Symbol"] + ".L"
        ftse100_tickers = ftse100_df["Symbol"].tolist()

        return {"SP500": sp500_tickers, "FTSE100": ftse100_tickers}
    except Exception as e:
        st.error(f"⚠️ Error loading stock symbols: {e}")
        return {"SP500": [], "FTSE100": []}

# 🔹 Get cached stock data from Firestore
def get_cached_stock_data(index_name):
    """
    index_name should be either 'sp500' or 'ftse100'
    Returns a DataFrame with reordered columns: Company Name, Symbol, etc.
    """
    try:
        doc_ref = db.collection("stock_cache").document(index_name)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if "stocks" in data:
                df = pd.DataFrame(data["stocks"])

                # 🔸 Desired column order
                preferred_order = ['CompanyName', 'Symbol', 'Price', 'PEratio', 'EPS', 'DividendYield', 'Market Cap', 'Sector']

                # 🔸 Reorder columns if present, keep any extra columns at the end
                ordered_cols = [col for col in preferred_order if col in df.columns]
                df = df[ordered_cols]

                return df
            else:
                st.warning(f"⚠️ No 'stocks' field found in Firestore document {index_name}.")
                return pd.DataFrame()
        else:
            st.warning(f"⚠️ Document {index_name} does not exist in Firestore.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching stock data from Firestore document {index_name}: {e}")
        return pd.DataFrame()

# Top gainers and losers cache

# 🔹 FMP API keys
API_KEY = st.secrets["FIREBASE"]["fmp_api_key"]
CACHE_DURATION_HOURS = 0.5  # Set cache expiry time to 12 hours

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

def get_most_actives():
    return fetch_and_cache_fmp_data("actives")
