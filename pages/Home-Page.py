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
from utils import get_top_gainers, get_top_losers, get_cached_stock_data

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

st.title('Welcome to InvestoPred!')

st.write('Please browse through the pages and try the functions!')


# Displaying Top Gainers and Losers
st.title("📈 Market Movers")

# Fetch top gainers and losers (this will use cached data if available)
gainers = get_top_gainers()
losers = get_top_losers()

# Display top 10 gainers
df_gainers = pd.DataFrame(gainers)[["symbol", "name", "price", "changesPercentage", "change"]].head(10)
df_gainers.columns = ["Symbol", "Company", "Price", "% Change", "Change ($)"]
st.subheader("Top 10 Gainers")
st.dataframe(df_gainers, use_container_width=True)

# Display top 10 losers
df_losers = pd.DataFrame(losers)[["symbol", "name", "price", "changesPercentage", "change"]].head(10)
df_losers.columns = ["Symbol", "Company", "Price", "% Change", "Change ($)"]
st.subheader("Top 10 Losers")
st.dataframe(df_losers, use_container_width=True)




# Load and cache stock data when user first visits homepage
if "SP500_data" not in st.session_state:
    st.session_state["SP500_data"] = get_cached_stock_data("SP500")

if "FTSE100_data" not in st.session_state:
    st.session_state["FTSE100_data"] = get_cached_stock_data("FTSE100")
