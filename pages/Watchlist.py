import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import time
import matplotlib.pyplot as plt
from navigation import navigation
from auth import track_session_activity
from tensorflow.keras.models import load_model

track_session_activity()

# Hide Streamlit's default multipage navigation
hide_streamlit_style = """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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


# Initialize session state
if "user" not in st.session_state:
    st.session_state["user"] = None
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Attempt to restore session
if not st.session_state["authenticated"]:
    try:
        session_docs = db.collection("sessions").stream()
        for doc in session_docs:
            session_data = doc.to_dict()
            if session_data.get("authenticated", False):
                st.session_state["authenticated"] = True
                st.session_state["user"] = {"email": doc.id}
                break
    except Exception as e:
        st.error(f"⚠️ Failed to restore session: {e}")

if not st.session_state.get("authenticated", False):
    st.error("Please log in to view this page.")
    st.sidebar.title("📌 Navigation")
    st.sidebar.page_link("pages/Home-Page.py", label="🏠 Home Page")
    st.sidebar.page_link("pages/Stock-Chart&Data.py", label="📈 Stock Charts and Data")
    st.sidebar.page_link("pages/loginsignup.py", label="🔑 Login / Signup")
    st.stop()

navigation()

# Load LSTM model
model = load_model('predict_model.keras')

# Prediction logic
def predict_next_90_days(hist, model):
    close_prices = hist["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)
    last_sequence = scaled_prices[-60:]
    future_predictions = []

    for _ in range(90):
        input_seq = last_sequence.reshape(1, 60, 1)
        pred = model.predict(input_seq, verbose=0)
        pred = np.maximum(pred, 0)
        future_predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)

    future_scaled = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_scaled)
    future_dates = pd.bdate_range(hist.index[-1] + pd.Timedelta(days=1), periods=90)
    return future_dates, future_prices

# Load tickers
def get_tickers(filepath):
    base_path = os.path.dirname(__file__)
    return pd.read_csv(filepath)["Symbol"].tolist()

sp500_tickers = get_tickers("sp500_companies.csv")
ftse100_tickers = get_tickers("FTSE100_Constituents.csv")
all_tickers = sorted(set(sp500_tickers + ftse100_tickers))

# Watchlist functionality
user_email = st.session_state["user"]["email"]
user_doc = db.collection("users").document(user_email)
watchlists_ref = user_doc.collection("watchlists")

st.title("📊 Your Watchlists")

# Sidebar actions
st.sidebar.header("🛠 Watchlist Actions")
watchlist_names = [doc.id for doc in watchlists_ref.stream()]

# Auto-create a default watchlist if user has none
if not watchlist_names:
    default_name = "My First Watchlist"
    watchlists_ref.document(default_name).set({"tickers": []})
    watchlist_names.append(default_name)
    st.success("✅ A default watchlist 'My First Watchlist' has been created for you.")
    time.sleep(1)
    st.rerun()

selected_watchlist = st.sidebar.selectbox("📋 Select a watchlist", watchlist_names)
st.session_state["selected_watchlist"] = selected_watchlist if selected_watchlist else None

# Create new watchlist
new_name = st.sidebar.text_input("🆕 Create new watchlist")
if st.sidebar.button("➕ Create") and new_name:
    watchlists_ref.document(new_name).set({"tickers": []})
    st.success(f"Watchlist '{new_name}' created!")
    st.rerun()

# Rename selected watchlist
if selected_watchlist:
    rename_input = st.sidebar.text_input("✏️ Rename watchlist")
    if st.sidebar.button("🔄 Rename") and rename_input:
        tickers_data = watchlists_ref.document(selected_watchlist).get().to_dict()
        watchlists_ref.document(rename_input).set(tickers_data)
        watchlists_ref.document(selected_watchlist).delete()
        st.success(f"Renamed to '{rename_input}'")
        st.rerun()

    # Delete selected watchlist
    if st.sidebar.button("🗑 Delete Watchlist"):
        watchlists_ref.document(selected_watchlist).delete()
        st.success(f"Watchlist '{selected_watchlist}' deleted.")
        st.rerun()

# Load selected watchlist
selected_tickers = []
if selected_watchlist:
    selected_doc = watchlists_ref.document(selected_watchlist).get()
    if selected_doc.exists:
        selected_tickers = selected_doc.to_dict().get("tickers", [])

# Add stock to watchlist
st.subheader(f"📈 Watchlist: {selected_watchlist}")
with st.expander("➕ Add stock to this watchlist"):
    new_ticker = st.selectbox("Select a ticker to add", [t for t in all_tickers if t not in selected_tickers])
    if st.button("Add to Watchlist"):
        selected_tickers.append(new_ticker)
        watchlists_ref.document(selected_watchlist).update({"tickers": selected_tickers})
        st.success(f"{new_ticker} added to {selected_watchlist}!")
        st.rerun()

# Remove stock
if selected_tickers:
    with st.expander("🗑 Remove stock from this watchlist"):
        remove_ticker = st.selectbox("Select a ticker to remove", selected_tickers)
        if st.button("Remove from Watchlist"):
            selected_tickers.remove(remove_ticker)
            watchlists_ref.document(selected_watchlist).update({"tickers": selected_tickers})
            st.success(f"{remove_ticker} removed.")
            st.rerun()

# Show charts
for ticker in selected_tickers:
    st.subheader(f"{ticker} Stock Overview")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info

        if hist.empty:
            st.warning(f"No data for {ticker}.")
            continue

        # Basic data
        name = info.get("shortName", "N/A")
        sector = info.get("sector", "N/A")
        high_52w = info.get("fiftyTwoWeekHigh", None)
        low_52w = info.get("fiftyTwoWeekLow", None)
        current_price = hist["Close"][-1]

        # Calculate price change if market open
        price_change = None
        if len(hist) >= 2:
            prev_close = hist["Close"][-2]
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

        # Distance from high/low
        from_high = ((high_52w - current_price) / high_52w * 100) if high_52w else None
        from_low = ((current_price - low_52w) / low_52w * 100) if low_52w else None

        # Show details
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Sector:** {sector}")
        st.markdown(f"**52-Week High:** ${high_52w:,.2f}" if high_52w else "**52-Week High:** N/A")
        st.markdown(f"**52-Week Low:** ${low_52w:,.2f}" if low_52w else "**52-Week Low:** N/A")
        st.markdown(f"**Price:** ${current_price:,.2f}")

        if price_change is not None:
            st.markdown(
                f"**Today's Change:** {'🔺' if price_change > 0 else '🔻'} ${price_change:.2f} "
                f"({price_change_pct:.2f}%)"
            )

        if from_high is not None and from_low is not None:
            st.markdown(
                f"**Retracement from 52W High:** {from_high:.2f}% ↓  |  "
                f"**Above 52W Low:** {from_low:.2f}% ↑"
            )

        # --- Historical Chart Only ---
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(hist.index, hist["Close"], label="Actual Prices", color='blue')
        ax.set_title(f"{ticker} Historical Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --- Prediction Button ---
        if st.button(f"Run Prediction for {ticker}"):
            with st.spinner("Running prediction..."):
                future_dates, future_prices = predict_next_90_days(hist, model)

            fig2, ax2 = plt.subplots(figsize=(14, 4))
            ax2.plot(hist.index, hist["Close"], label="Actual Prices", color='blue')
            ax2.plot(future_dates, future_prices.flatten(), label="Predicted Prices", color='red')
            ax2.set_title(f"{ticker} Price Forecast (Next 90 Days)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error with {ticker}: {e}")


st.markdown("""
<hr style="margin-top: 2em;">
<small>
📜 **Disclaimer:** This app is for informational purposes only and does not constitute financial advice. Always do your own research or consult a professional.
</small>
""", unsafe_allow_html=True)
