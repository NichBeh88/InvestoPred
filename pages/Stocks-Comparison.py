import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
from navigation import navigation
from auth import track_session_activity

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

st.title("📊 Stock Comparisons & Predictions")
st.caption("**What can you do here?**", help="Select the stock index(es) that you want to compare from the Navigation Bar beside. Then, click the dropout column to select or type in your interested stock(s) symbol and click *Compare & Predict Stocks*.")

# Load stock symbols based on index
@st.cache_data
def load_stock_symbols(index="sp500"):
    if index == "sp500":
        df = pd.read_csv("sp500_companies.csv")
    else:
        df = pd.read_csv("FTSE100_Constituents.csv")
        df["Symbol"] = df["Symbol"] + ".L"  # Convert to Yahoo Finance format

    return df["Symbol"].tolist()

# Sidebar for index selection
index_choice1 = st.sidebar.radio("Select First Stock Index", ["S&P 500", "FTSE 100"])
index_choice2 = st.sidebar.radio("Select Second Stock Index", ["S&P 500", "FTSE 100"])

symbols1 = load_stock_symbols(index="sp500" if index_choice1 == "S&P 500" else "ftse100")
symbols2 = load_stock_symbols(index="sp500" if index_choice2 == "S&P 500" else "ftse100")

# Select two stocks for comparison
st.subheader("📊 Compare Two Stocks")
col1, col2 = st.columns(2)
stock1 = col1.selectbox("Select First Stock", symbols1, key="stock1")
stock2 = col2.selectbox("Select Second Stock", symbols2, key="stock2")

def get_financial_data(stock):
    """Fetch financial data for a stock"""
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info
        return {
            "Stock": stock,
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "Market Cap": f"{info.get('marketCap', 'N/A'):,}" if "marketCap" in info else "N/A",
            "Dividend Yield": f"{info.get('dividendYield', 0) :.2f}%" if "dividendYield" in info else "N/A",
            "Revenue": f"{info.get('totalRevenue', 'N/A'):,}" if "totalRevenue" in info else "N/A",
            "Net Income": f"{info.get('netIncomeToCommon', 'N/A'):,}" if "netIncomeToCommon" in info else "N/A"
        }
    except Exception as e:
        st.error(f"Error fetching data for {stock}: {e}")
        return None

if st.button("Compare & Predict Stocks"):
    with st.status(f"{stock1} and {stock2}. Got it!! Working on it...!!", expanded=False) as status:

        # Get financial data
        stock1_data = get_financial_data(stock1)
        stock2_data = get_financial_data(stock2)
    
        if stock1_data and stock2_data:
            df = pd.DataFrame([stock1_data, stock2_data])
            st.subheader("📊 Financial Comparison")
            st.dataframe(df.set_index("Stock"))
    
        # Fetch historical data
        data1 = yf.download(stock1, period="5y")
        data2 = yf.download(stock2, period="5y")
    
        if not data1.empty and not data2.empty:
            # Plot past stock trends
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data1.index, data1['Close'], label=f"{stock1} (Blue)", color='blue')
            ax.plot(data2.index, data2['Close'], label=f"{stock2} (Red)", color='red')
            ax.set_title(f"Stock Price Trend: {stock1} vs {stock2}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
    
            # Stock 1
            st.subheader(f"📈 {stock1} - Historical Prices")
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(data1.index, data1['Close'], color='blue', label=f"{stock1}")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price")
            ax1.legend()
            st.pyplot(fig1)
    
            # Stock 2
            st.subheader(f"📈 {stock2} - Historical Prices")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(data2.index, data2['Close'], color='red', label=f"{stock2}")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price")
            ax2.legend()
            st.pyplot(fig2)
    
            # Load prediction model
            model = load_model('predict_model.keras')
    
            def predict_stock(stock, data):
                prices = data['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(prices)
    
                time_step = 60
                last_sequence = scaled_prices[-time_step:]
                future_predictions = []
    
                for _ in range(90):
                    last_sequence_reshaped = last_sequence.reshape(1, time_step, 1)
                    next_price = model.predict(last_sequence_reshaped)
                    next_price = np.maximum(next_price, 0)
                    future_predictions.append(next_price[0, 0])
                    last_sequence = np.append(last_sequence[1:], next_price, axis=0)
    
                return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
            # Predict stock prices for next 90 days
            future_predictions1 = predict_stock(stock1, data1)
            future_predictions2 = predict_stock(stock2, data2)
            future_dates = pd.date_range(start=data1.index[-1] + timedelta(days=1), periods=90)
    
            # Plot future stock predictions
            st.subheader("📅 Next 90 Days Forecast")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(future_dates, future_predictions1, color='blue', label=f"{stock1} Prediction")
            ax.plot(future_dates, future_predictions2, color='red', label=f"{stock2} Prediction")
            ax.set_title(f"{stock1} vs {stock2} - Predicted Stock Prices (Next 90 Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)
    
            # Plot individual stock predictions
            col1, col2 = st.columns(2)
    
            with col1:
                st.subheader(f"📈 {stock1} Prediction (Next 90 Days)")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(future_dates, future_predictions1, color='blue', label=f"{stock1} Prediction")
                ax1.set_xlabel("Date")
                plt.xticks(rotation=45)
                ax1.set_ylabel("Price")
                ax1.legend()
                st.pyplot(fig1)
    
            with col2:
                st.subheader(f"📈 {stock2} Prediction (Next 90 Days)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(future_dates, future_predictions2, color='red', label=f"{stock2} Prediction")
                ax2.set_xlabel("Date")
                plt.xticks(rotation=45)
                ax2.set_ylabel("Price")
                ax2.legend()
                st.pyplot(fig2)

        else:
            st.warning("⚠️ Failed to fetch stock data for one or both stocks.")
            status.update(label="❌ Failed to fetch data", state="error", expanded=True)
    
    status.update(label="✅ Done!", state="complete", expanded=True)

st.markdown("""
<hr style="margin-top: 2em;">
<small>
📜 **Disclaimer:** This app is for informational purposes only and does not constitute financial advice. Always do your own research or consult a professional.
</small>
""", unsafe_allow_html=True)
