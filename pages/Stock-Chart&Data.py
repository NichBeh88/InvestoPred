import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from navigation import navigation
import firebase_admin
from firebase_admin import credentials, firestore
from auth import track_session_activity

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

# Streamlit UI
st.title("📊 Stock Selection & Prediction with Candlestick Chart & Indicators")

# Load stock symbols for S&P 500 and FTSE 100
@st.cache_data
def load_stock_symbols(index="sp500"):
    if index == "sp500":
        df = pd.read_csv("sp500_companies.csv")
    else:
        df = pd.read_csv("FTSE100_Constituents.csv")
        df["Symbol"] = df["Symbol"] + ".L"

    return df["Symbol"].tolist()

navigation()

# Redirect to Home Page if logout was triggered
if st.session_state.get("logout_triggered", False):
    st.session_state["logout_triggered"] = False  # Reset trigger
    st.switch_page("pages/Home-Page.py")  # Redirect immediately

index_choice = st.sidebar.radio("Select Stock Index", ["S&P 500", "FTSE 100"])
symbols = load_stock_symbols(index="sp500" if index_choice == "S&P 500" else "ftse100")
selected_stock = st.sidebar.selectbox("Select a Stock", symbols)

# Fetch fresh stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        stock_history = stock.history(period="5y")

        return {
            "Symbol": symbol,
            "Company Name": info.get("shortName", "N/A"),
            "Sector": info.get("sector", "Unknown"),
            "P/E Ratio": info.get("trailingPE", np.nan),
            "Dividend Yield (%)": info.get("dividendYield", 0),
            "Market Cap": info.get("marketCap", np.nan),
            "EPS": info.get("trailingEps", np.nan),
            "Price": info.get("previousClose", np.nan),
            "Price History": stock_history
        }
    except Exception as e:
        st.error(f"⚠️ Error fetching {symbol}: {e}")
        return None

# Load stock data
stock_data = fetch_stock_data(selected_stock)

if stock_data is None:
    st.stop()

# Display stock data
st.subheader(f"📈 {stock_data['Company Name']} ({selected_stock})")
st.write(f"**Sector**: {stock_data['Sector']}")
st.write(f"**P/E Ratio**: {stock_data['P/E Ratio']}")
st.write(f"**Dividend Yield**: {stock_data['Dividend Yield (%)']}%")
st.write(f"**Market Cap**: {stock_data['Market Cap']}")
st.write(f"**EPS**: {stock_data['EPS']}")
st.write(f"**Price**: {stock_data['Price']}")

# Prepare stock price chart
df = stock_data["Price History"]
df['Date'] = df.index
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=(f"{selected_stock} Candlestick Chart", "RSI (14)"),
                    row_heights=[0.8, 0.2])

fig.add_trace(go.Candlestick(x=df['Date'],
                                    open=df['Open'],
                                    high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    name='Candlesticks'), row=1, col=1)

fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], line=dict(color='blue', width=2), name='SMA 50'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], line=dict(color='red', width=2), name='SMA 200'))

fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='orange', width=2), name='RSI (14)'), row=2, col=1)
fig.add_hline(y=80, line=dict(color='red', dash='dash'), row=2, col=1)
fig.add_hline(y=20, line=dict(color='green', dash='dash'), row=2, col=1)

fig.update_layout(
    title=f"{selected_stock} Candlestick Chart with Indicators",
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    template='plotly_dark',
    hovermode='closest',
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig)
