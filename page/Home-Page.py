import streamlit as st
from navigation import navigation
import firebase_admin
from firebase_admin import credentials, firestore
from auth import track_session_activity
from yahooquery import Screener
import pandas as pd

track_session_activity()

# Initialize Firebase
if not firebase_admin._apps:
    firebase_creds = st.secrets["FIREBASE"]
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()
FIREBASE_API_KEY = st.secrets["api_key"]

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

st.title('Welcome to InvestoPred!')

st.write('Please browse through the pages and try the functions!')

st.session_state["current_page"] = "Home-Page"  # Track the page


# Function to fetch top gainers from Yahoo Finance
def get_top_gainers():
    screener = Screener()
    
    try:
        # Fetch "day_gainers" from Yahoo Finance
        data = screener.get_screeners("day_gainers", count=10)  # Get top 10 gainers

        # Extract relevant stock data
        stocks = data["day_gainers"]["quotes"]
        stock_list = [
            {
                "Symbol": stock["symbol"],
                "Price": float(stock["regularMarketPrice"]) if stock.get("regularMarketPrice") else 0.0,
                "Change %": float(stock["regularMarketChangePercent"]) if stock.get("regularMarketChangePercent") else 0.0,
                "Dividend Yield (%)": float(stock.get("dividendYield", 0)) if stock.get("dividendYield") else 0.0,
                "Volume": int(stock["regularMarketVolume"]) if stock.get("regularMarketVolume") else 0
            }
            for stock in stocks
        ]

        # Convert to DataFrame for better display
        df = pd.DataFrame(stock_list)

        # Ensure all numbers have 2 decimal places
        df["Change %"] = df["Change %"].apply(lambda x: round(x, 2))
        df["Dividend Yield (%)"] = df["Dividend Yield (%)"].apply(lambda x: round(x, 2))
        df["Price"] = df["Price"].apply(lambda x: round(x, 2))

        # Sort by highest change percentage
        return df.sort_values(by="Change %", ascending=False)  # Sort by highest gain

    except Exception as e:
        st.write(f"❌ Error fetching top gainers: {e}")
        return pd.DataFrame()

# Fetch and display top gainers
top_gainers_df = get_top_gainers()
if not top_gainers_df.empty:
    st.write("\n📈 Top Market Gainers Today:\n")
    st.table(top_gainers_df)
else:
    st.write("No data available.")


# top losers
def get_top_losers():
    screener = Screener()
    
    try:
        # Fetch "day_losers" from Yahoo Finance
        data = screener.get_screeners("day_losers", count=10)  # Get top 10 losers

        # Extract relevant stock data
        stocks = data["day_losers"]["quotes"]
        stock_list = [
            {
                "Symbol": stock["symbol"],
                "Price": float(stock["regularMarketPrice"]) if stock.get("regularMarketPrice") else 0.0,
                "Change %": float(stock["regularMarketChangePercent"]) if stock.get("regularMarketChangePercent") else 0.0,
                "Dividend Yield (%)": float(stock.get("dividendYield", 0)) if stock.get("dividendYield") else 0.0,
                "Volume": int(stock["regularMarketVolume"]) if stock.get("regularMarketVolume") else 0
            }
            for stock in stocks
        ]

        # Convert to DataFrame for better display
        df = pd.DataFrame(stock_list)

        # Ensure all numbers have 2 decimal places
        df["Change %"] = df["Change %"].apply(lambda x: round(x, 2))
        df["Dividend Yield (%)"] = df["Dividend Yield (%)"].apply(lambda x: round(x, 2))
        df["Price"] = df["Price"].apply(lambda x: round(x, 2))

        # Sort by highest negative change percentage
        return df.sort_values(by="Change %", ascending=True)  # Sort by highest loss

    except Exception as e:
        st.write(f"❌ Error fetching top losers: {e}")
        return pd.DataFrame()

# Fetch and display top losers
top_losers_df = get_top_losers()
if not top_losers_df.empty:
    st.write("\n📉 Top Market Losers Today:\n")
    st.table(top_losers_df)
else:
    st.write("No data available.")



from utils import get_cached_stock_data

# Load and cache stock data when user first visits homepage
if "SP500_data" not in st.session_state:
    st.session_state["SP500_data"] = get_cached_stock_data("SP500")

if "FTSE100_data" not in st.session_state:
    st.session_state["FTSE100_data"] = get_cached_stock_data("FTSE100")



