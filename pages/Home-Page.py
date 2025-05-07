import streamlit as st
import sys
import os

# ✅ Ensure root directory is in sys.path so we can import utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from navigation import navigation
import firebase_admin
from firebase_admin import credentials, firestore
from auth import track_session_activity
import pandas as pd
from utils import get_top_gainers, get_top_losers, get_cached_stock_data, get_most_actives

track_session_activity()

# Check if Firebase has already been initialized
if not firebase_admin._apps:
    # Access the private key and other credentials from Streamlit secrets
    private_key = st.secrets["FIREBASE"]["private_key"]
    client_email = st.secrets["FIREBASE"]["client_email"]
    project_id = st.secrets["FIREBASE"]["project_id"]
    
    # Create a dictionary with the credentials
    service_account_info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key": private_key,
        "client_email": client_email,
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40investopred.iam.gserviceaccount.com"
    }
    
    # Initialize Firebase with the credentials
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
    
else:
    print("Firebase app is already initialized.")

db = firestore.client()
FIREBASE_API_KEY = st.secrets["FIREBASE"]["api_key"]

# 🛡 Initialize session keys safely
if "user" not in st.session_state:
    st.session_state["user"] = None
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# 🔍 Attempt to restore session from Firestore if not authenticated but session exists in DB
if not st.session_state["authenticated"]:
    try:
        # Check if email exists in session (e.g. stored in local storage previously)
        # In full production you'd pass session id in cookies or similar
        session_docs = db.collection("sessions").stream()
        for doc in session_docs:
            session_data = doc.to_dict()
            if session_data.get("authenticated", False):
                st.session_state["authenticated"] = True
                st.session_state["user"] = {"email": doc.id}
                break
    except Exception as e:
        st.error(f"⚠ Failed to restore session: {e}")

# Hide Streamlit's default multipage navigation
hide_streamlit_style = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Navigation menu
navigation()

# Redirect to Home Page if logout was triggered
if st.session_state.get("logout_triggered", False):
    st.session_state["logout_triggered"] = False  # Reset trigger
    st.switch_page("pages/Home-Page.py")  # Redirect immediately

st.markdown("""
### 🚀 *Predict. Analyze. Succeed.*
Welcome to **InvestoPred** – your all-in-one platform for stock market prediction, analysis, and insights.  
""")
st.markdown("### 🔍 What You Can Do Here:")
st.markdown("""
- 📊 **Visualize Stock Trends** – Interactive charts and indicators.
- 🤖 **Predict Future Prices** – AI-powered stock predictions.
- 🔔 **Create Custom Watchlists** – Stay updated with your favorite stocks.
- 📈 **Compare Stocks** – Evaluate performance side-by-side.
""")
st.markdown("📬 Sign up an account for free to access all features!")

# Displaying Top Gainers and Losers
st.title("💰 Today's Market Movers")

# Fetch top gainers and losers (this will use cached data if available)
gainers = get_top_gainers()
losers = get_top_losers()
actives = get_most_actives()

# Display top 10 gainers
df_gainers = pd.DataFrame(gainers)[["symbol", "name", "price", "changesPercentage", "change"]].head(10)
df_gainers.columns = ["Symbol", "Company", "Price", "% Change", "Change ($)"]
st.subheader("📈Top 10 Gainers")
st.dataframe(df_gainers, use_container_width=True)

# Display top 10 losers
df_losers = pd.DataFrame(losers)[["symbol", "name", "price", "changesPercentage", "change"]].head(10)
df_losers.columns = ["Symbol", "Company", "Price", "% Change", "Change ($)"]
st.subheader("📉Top 10 Losers")
st.dataframe(df_losers, use_container_width=True)

# Display top 10 active
df_active = pd.DataFrame(actives)[["symbol", "name", "price", "changesPercentage", "change"]].head(10)
df_active.columns = ["Symbol", "Company", "Price", "% Change", "Change ($)"]
st.subheader("🔥 Most Actively Traded Stocks")
st.dataframe(df_active, use_container_width=True)

# --- Personal Watchlist (Authenticated Users Only) ---
if st.session_state.get("authenticated", False):
    user_email = st.session_state["user"]["email"]
    user_doc = db.collection("users").document(user_email)
    watchlists_ref = user_doc.collection("watchlists")

    # Get all watchlists
    watchlists = watchlists_ref.stream()
    watchlist_data = {doc.id: doc.to_dict().get("tickers", []) for doc in watchlists}
    
    if watchlist_data:
        # Pick the first watchlist
        first_watchlist_name = list(watchlist_data.keys())[0]
        tickers = watchlist_data[first_watchlist_name]

        st.subheader(f"📌 Watchlist: {first_watchlist_name}")

        if tickers:
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="2d")
                    if hist.empty or len(hist) < 2:
                        continue
                    today_price = hist["Close"].iloc[-1]
                    yesterday_price = hist["Close"].iloc[-2]
                    change_pct = ((today_price - yesterday_price) / yesterday_price) * 100

                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{ticker}**")
                    with col2:
                        st.markdown(f"Price: ${today_price:.2f}")
                    with col3:
                        change_color = "green" if change_pct >= 0 else "red"
                        st.markdown(f"<span style='color:{change_color}'>Change: {change_pct:.2f}%</span>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"⚠️ Error loading {ticker}: {e}")
        else:
            st.info("This watchlist is empty.")
    else:
        st.info("You have no watchlists yet. Create one from the Watchlist page.")
else:
    st.info("Please log in to view your watchlist.")
    
# Load and cache stock data when user first visits homepage
if "SP500_data" not in st.session_state:
    st.session_state["SP500_data"] = get_cached_stock_data("sp500")

if "FTSE100_data" not in st.session_state:
    st.session_state["FTSE100_data"] = get_cached_stock_data("ftse100")
