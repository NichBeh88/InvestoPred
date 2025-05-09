import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from utils import get_cached_stock_data  # ✅ Import from utils.py
from navigation import navigation
import firebase_admin
from firebase_admin import credentials, firestore
from auth import track_session_activity

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

track_session_activity()

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

# Check if the user is authenticated before showing the page content
if not st.session_state.get("authenticated", False):
    st.error("Please log in to view this page.")
    st.sidebar.title("📌 Navigation")
    st.sidebar.page_link("pages/Home-Page.py", label="🏠 Home Page")
    st.sidebar.page_link("pages/Stock-Chart&Data.py", label="📈 Stock Charts and Data")
    st.sidebar.page_link("pages/loginsignup.py", label="🔑 Login / Signup")
    st.stop()

navigation()

# Redirect to Home Page if logout was triggered
if st.session_state.get("logout_triggered", False):
    st.session_state["logout_triggered"] = False  # Reset trigger
    st.switch_page("pages/Home-Page.py")  # Redirect immediately

# ✅ Fetch cached stock data (from utils.py)
financial_df = get_cached_stock_data("ftse100")

# Streamlit UI Setup
st.title("🇬🇧 FTSE100 Stock Screener & Prediction")
st.caption("**What can you do here?** ℹ️", help="Try out the filter functions in side Navigation Bar to filter out the stock(s) that fits your criteria(s) the most! Then, scroll down and try out our model to predict the potential next 90 days trend of the stock(s) filtered!")

# Sidebar Filters for stock selection
st.sidebar.header("🔍 Filter Stocks")
price_min = st.sidebar.number_input("Min Price (£)", value=np.nan)
price_max = st.sidebar.number_input("Max Price (£)", value=np.nan)
pe_min = st.sidebar.number_input("Min P/E", value=np.nan)
pe_max = st.sidebar.number_input("Max P/E", value=np.nan)
dy_min = st.sidebar.number_input("Min Dividend Yield (%)", value=np.nan)
eps_min = st.sidebar.number_input("Min EPS (£)", value=np.nan)
sector = st.sidebar.selectbox("Sector", ["All"] + list(financial_df["Sector"].dropna().unique()))

# Apply Filters
filtered_stocks = financial_df.copy()
if not np.isnan(price_min):
    filtered_stocks = filtered_stocks[filtered_stocks["Price"] >= price_min]
if not np.isnan(price_max):
    filtered_stocks = filtered_stocks[filtered_stocks["Price"] <= price_max]
if not np.isnan(pe_min):
    filtered_stocks = filtered_stocks[filtered_stocks["PEratio"] >= pe_min]
if not np.isnan(pe_max):
    filtered_stocks = filtered_stocks[filtered_stocks["PEratio"] <= pe_max]
if not np.isnan(dy_min):
    filtered_stocks = filtered_stocks[filtered_stocks["DividendYield"] >= dy_min]
if not np.isnan(eps_min):
    filtered_stocks = filtered_stocks[filtered_stocks["EPS"] >= eps_min]
if sector != "All":
    filtered_stocks = filtered_stocks[filtered_stocks["Sector"] == sector]

st.subheader("📊 Filtered Stocks")
st.caption("**Keyterms** ℹ️", help="EPS: Earnings Per Share, calulated by dividing a company's net income by total number of outstanding shares. PEratio: Price-To-Earnings Ratio, calculated by dividing company's share price to earnings per share. For FTSE100 stocks, EPS needs to x100 before dividing as price for each company's shares are calculated by lots and 1 lot size = 100 units.")
st.dataframe(filtered_stocks)
st.write(f"Total Stocks Matched: **{len(filtered_stocks)}**")

# Stock Prediction Section
st.subheader("📊 Stock Price Prediction")
st.caption("**Guide** ℹ️", help="Choose the stock that you want to predict from the dropout column and click the *Predict Price* button. Then, just wait for the model to do the rest and see the prediction outcome!")
selected_stock = st.selectbox("Select a stock for prediction:", filtered_stocks["Symbol"].tolist())

if st.button("Predict Price"):
    with st.status(f"{selected_stock}, Alohomora!!", expanded=False) as status:
        data = yf.download(selected_stock, start='2001-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        prices = data['Close'].values.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # Load pre-trained LSTM model
        model = load_model('predict_model.keras') 
        
        # Predict next 90 days
        time_step = 60
        last_sequence = scaled_prices[-time_step:]
        future_predictions = []
        for _ in range(90):
            last_sequence_reshaped = last_sequence.reshape(1, time_step, 1)
            next_price = model.predict(last_sequence_reshaped)
            next_price = np.maximum(next_price, 0)
            future_predictions.append(next_price[0, 0])
            last_sequence = np.append(last_sequence[1:], next_price, axis=0)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=90)
        
        # Plot Predictions
        st.subheader("📅 Next 90 Days Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(future_dates, future_predictions, color='green', label='Predicted Price (£)')
        ax.set_title(f"{selected_stock} - Predicted Stock Price (Next 90 Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (£)")
        ax.legend()
        st.pyplot(fig)
        
        st.success("Prediction Complete!")
        st.write("Download predictions below:")
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price (£)": future_predictions.flatten()})
        st.download_button(label="Download CSV", data=forecast_df.to_csv(index=False), file_name=f"{selected_stock}_forecast.csv", mime='text/csv')

        status.update(label="✅ Done!", state="complete", expanded=True)

st.markdown("""
<hr style="margin-top: 2em;">
<small>
📜 **Disclaimer:** This app is for informational purposes only and does not constitute financial advice. Always do your own research or consult a professional.
</small>
""", unsafe_allow_html=True)
