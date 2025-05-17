from flask import Flask, render_template, request, redirect, url_for, session
from auth import (
    init_firebase,
    get_user_from_session_cookie,
    verify_password_and_login,
    create_user  # ✅ Must be imported!
)
import os
from dotenv import load_dotenv

# 🔄 Load .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# 🔐 Initialize Firebase
init_firebase()

@app.route("/")
def home():
    user = get_user_from_session_cookie()
    return render_template("home.html", user=user)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if verify_password_and_login(email, password):
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid email or password")
    return render_template("login.html", error=None)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        success, error = create_user(email, password)
        if success:
            return render_template("signup.html", success="Account created! You can now log in.", error=None)
        else:
            return render_template("signup.html", error=error, success=None)
    return render_template("signup.html", error=None, success=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/chart", methods=["GET", "POST"])
def chart():
    user = get_user_from_session_cookie()
    symbol = None

    if request.method == "POST":
        raw = request.form["symbol"].strip().upper()
        if raw.endswith(".L"):
            symbol = f"LSE:{raw}"
        elif raw in ["TSLA", "AAPL", "NVDA", "MSFT", "GOOGL"]:
            symbol = f"NASDAQ:{raw}"
        else:
            symbol = raw  # fallback

    return render_template("chart.html", user=user, symbol=symbol)



@app.route("/watchlist", methods=["GET", "POST"])
def watchlist():
    from firebase_admin import firestore
    db = firestore.client()

    user = get_user_from_session_cookie()
    if not user:
        return redirect(url_for("login"))

    user_id = user["email"]
    user_ref = db.collection("users").document(user_id)
    watchlist_ref = user_ref.collection("watchlists")

    # Determine selected watchlist and index filter
    selected_list = request.args.get("selected_list") or request.form.get("list_name")
    index_filter = request.args.get("index_filter") or request.form.get("index_filter") or "all"

    # Load screener index data
    sp500_doc = db.collection("screener_index").document("sp500").get().to_dict() or {}
    ftse_doc = db.collection("screener_index").document("ftse100").get().to_dict() or {}

    # Filter stocks based on index selection
    sp500_stocks = sp500_doc.get("stocks", [])
    ftse_stocks = ftse_doc.get("stocks", [])

    if index_filter == "sp500":
        all_stocks = sp500_stocks
    elif index_filter == "ftse100":
        all_stocks = ftse_stocks
    else:
        all_stocks = sp500_stocks + ftse_stocks

    # POST handling
    if request.method == "POST":
        action = request.form.get("action")
        new_name = request.form.get("new_name", "").strip()
        create_name = request.form.get("create_name", "").strip()
        symbol = request.form.get("symbol", "").strip().upper()

        if action == "create_list" and create_name:
            selected_list = create_name
            doc_ref = watchlist_ref.document(create_name)
            if not doc_ref.get().exists:
                doc_ref.set({"symbols": []})

        elif selected_list:
            doc_ref = watchlist_ref.document(selected_list)

            if action == "add":
                existing = doc_ref.get().to_dict() or {}
                symbols = existing.get("symbols", [])
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
                doc_ref.set({"symbols": symbols})

            elif action == "remove":
                existing = doc_ref.get().to_dict() or {}
                symbols = existing.get("symbols", [])
                if symbol in symbols:
                    symbols.remove(symbol)
                doc_ref.set({"symbols": symbols})

            elif action == "delete_list":
                doc_ref.delete()
                selected_list = None

            elif action == "rename_list" and new_name:
                old_doc = doc_ref.get().to_dict()
                if old_doc:
                    watchlist_ref.document(new_name).set(old_doc)
                    doc_ref.delete()
                    selected_list = new_name

    # Load user watchlists
    docs = watchlist_ref.stream()
    watchlists = {doc.id: doc.to_dict().get("symbols", []) for doc in docs}

    return render_template(
        "watchlist.html",
        user=user,
        watchlists=watchlists,
        selected_list=selected_list,
        all_stocks=all_stocks,
        index_filter=index_filter
    )




@app.route("/screener", methods=["GET", "POST"])
def screener():
    from firebase_admin import firestore
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    from datetime import datetime, timedelta
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    import base64
    import requests
    import os

    db = firestore.client()

    user = get_user_from_session_cookie()
    user_id = user["email"] if user else None
    user_watchlists = []

    if user_id:
        wl_ref = db.collection("users").document(user_id).collection("watchlists")
        user_watchlists = [doc.id for doc in wl_ref.stream()]

    # Filters
    index_filter = request.args.get("index_filter", "sp500")
    sector_filter = request.args.get("sector", "all")
    pe_min = request.args.get("pe_min", type=float)
    pe_max = request.args.get("pe_max", type=float)
    eps_min = request.args.get("eps_min", type=float)
    eps_max = request.args.get("eps_max", type=float)
    price_min = request.args.get("price_min", type=float)
    price_max = request.args.get("price_max", type=float)
    div_min = request.args.get("div_min", type=float)
    div_max = request.args.get("div_max", type=float)
    sort_key = request.args.get("sort", "Symbol")
    sort_dir = request.args.get("direction", "asc")

    # Load stocks from Firestore screener index
    index_doc = db.collection("screener_index").document(index_filter).get().to_dict() or {}
    stocks = index_doc.get("stocks", [])

    # Apply filters
    filtered = []
    for stock in stocks:
        try:
            symbol = stock.get("Symbol", "")
            sector = stock.get("Sector", "Unknown")
            pe = float(stock.get("PEratio", 0))
            eps = float(stock.get("EPS", 0))
            price = float(stock.get("Price", 0))
            div = float(stock.get("DividendYield", 0))

            if sector_filter != "all" and sector != sector_filter:
                continue
            if pe_min is not None and pe < pe_min:
                continue
            if pe_max is not None and pe > pe_max:
                continue
            if eps_min is not None and eps < eps_min:
                continue
            if eps_max is not None and eps > eps_max:
                continue
            if price_min is not None and price < price_min:
                continue
            if price_max is not None and price > price_max:
                continue
            if div_min is not None and div < div_min:
                continue
            if div_max is not None and div > div_max:
                continue

            filtered.append(stock)
        except:
            continue

    try:
        filtered.sort(key=lambda x: x.get(sort_key, 0), reverse=(sort_dir == "desc"))
    except:
        pass

    # ✅ Watchlist handler
    if request.method == "POST" and request.form.get("action") == "add_watchlist" and user_id:
        wl_name = request.form.get("watchlist_name")
        symbol = request.form.get("symbol")
        if wl_name and symbol:
            wl_ref = db.collection("users").document(user_id).collection("watchlists").document(wl_name)
            doc = wl_ref.get()
            if doc.exists:
                data = doc.to_dict()
                symbols = data.get("symbols", [])
                if symbol not in symbols:
                    symbols.append(symbol)
                    wl_ref.set({"symbols": symbols})
            else:
                wl_ref.set({"symbols": [symbol]})

    prediction_plot = None
    forecast_csv = None
    selected_symbol = None
    fetching_message = None

    # ✅ Prediction trigger
    if request.method == "POST" and request.form.get("predict_symbol"):
        selected_symbol = request.form.get("predict_symbol", "").strip().upper()
        data = {}

        if selected_symbol.endswith(".L"):
            # 🔹 FTSE100: Load from Firestore (already cached by daily Cloud Function)
            doc = db.collection("stock_cache").document(selected_symbol).get()
            if doc.exists:
                data = doc.to_dict().get("historical_data", {})
        else:
            # 🔸 S&P500: Fetch live from FMP
            FMP_API_KEY = os.getenv("FMP_API_KEY")
            fmp_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{selected_symbol}?serietype=line&apikey={FMP_API_KEY}&timeseries=2000"
            try:
                resp = requests.get(fmp_url)
                fmp_data = resp.json().get("historical", [])
                for row in fmp_data:
                    data[row["date"]] = {
                        "Close": row["close"],
                        "Open": row.get("open", row["close"]),
                        "High": row.get("high", row["close"]),
                        "Low": row.get("low", row["close"]),
                        "Volume": row.get("volume", 0)
                    }
            except Exception as e:
                fetching_message = f"⚠️ Failed to fetch from FMP: {e}"

        # ✅ Run prediction if data is valid
        if data and len(data) >= 1000:
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            if "Close" in df.columns:
                prices = df[["Close"]].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_prices = scaler.fit_transform(prices)

                time_step = 60
                last_sequence = scaled_prices[-time_step:]
                future_predictions = []

                model = load_model("PredictModel.keras")

                for _ in range(90):
                    input_seq = last_sequence.reshape(1, time_step, 1)
                    next_price = model.predict(input_seq, verbose=0)
                    next_price = np.maximum(next_price, 0)
                    future_predictions.append(next_price[0, 0])
                    last_sequence = np.append(last_sequence[1:], next_price, axis=0)

                future_scaled = np.array(future_predictions).reshape(-1, 1)
                future_prices = scaler.inverse_transform(future_scaled)

                future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=90)
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price ($)": future_prices.flatten()})

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(future_dates, future_prices, color='green')
                ax.set_title(f"{selected_symbol} - Predicted Stock Price (Next 90 Days)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.grid(True)

                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                prediction_plot = f"data:image/png;base64,{plot_data}"
                forecast_csv = forecast_df.to_csv(index=False)
        else:
            # Optional fallback to trigger fetch
            source = "yf" if selected_symbol.endswith(".L") else "fmp"
            try:
                cloud_url = f"https://us-central1-investopred.cloudfunctions.net/updateStockHistory?symbol={selected_symbol}&source={source}"
                requests.get(cloud_url)
                fetching_message = f"⏳ Fetching historical data for {selected_symbol} from {source.upper()}. Please retry in 30–60 seconds."
            except Exception as e:
                fetching_message = f"⚠️ Failed to trigger fetch from {source.upper()}: {e}"

    return render_template(
        "screener.html",
        stocks=filtered,
        sectors=sector_filter,
        index_filter=index_filter,
        sector_filter=sector_filter,
        pe_min=pe_min,
        pe_max=pe_max,
        eps_min=eps_min,
        eps_max=eps_max,
        price_min=price_min,
        price_max=price_max,
        div_min=div_min,
        div_max=div_max,
        sort_key=sort_key,
        sort_dir=sort_dir,
        selected_symbol=selected_symbol,
        prediction_plot=prediction_plot,
        forecast_csv=forecast_csv,
        fetching_message=fetching_message,
        user=user,
        user_watchlists=user_watchlists
    )




@app.route("/compare", methods=["GET", "POST"])
def compare():
    from firebase_admin import firestore
    db = firestore.client()

    user = get_user_from_session_cookie()


    index_a = request.form.get("index_a", "sp500")
    index_b = request.form.get("index_b", "sp500")
    stock_a = request.form.get("stock_a")
    stock_b = request.form.get("stock_b")

    def get_symbols(index):
        doc = db.collection("screener_index").document(index).get().to_dict() or {}
        return sorted([s.get("Symbol") for s in doc.get("stocks", []) if s.get("Symbol")])

    def get_fundamentals(symbol):
        indexes = ["sp500", "ftse100"]
        found = False
        stock_data = None

        for index in indexes:
            doc = db.collection("screener_index").document(index).get().to_dict() or {}
            for stock in doc.get("stocks", []):
                if stock.get("Symbol") == symbol:
                    stock_data = stock
                    found = True
                    break
            if found:
                break

        # Fallback if not found in Firestore
        if not stock_data:
            stock_data = {}

        # Handle exchange prefix fallback
        if symbol.endswith('.L'):
            exchange_symbol = symbol  # FTSE stock
        else:
            exchange_symbol = f"NASDAQ:{symbol}"  # Default try NASDAQ
            # You can later attempt fetching with NYSE if required (e.g., in frontend/JS if NASDAQ fails)

        return {
            "Symbol": exchange_symbol,
            "Sector": stock_data.get("Sector", "N/A"),
            "Market Cap": f"{stock_data.get('MarketCap', 'N/A'):,}" if "MarketCap" in stock_data else "N/A",
            "P/E Ratio": stock_data.get("PEratio", "N/A"),
            "EPS": stock_data.get("EPS", "N/A"),
            "Dividend Yield": f"{stock_data.get('DividendYield', 0):.2f}%",
            "Price": stock_data.get("Price", "N/A"),
            "Company": stock_data.get("CompanyName", "N/A")
        }


    def compare_values(val1, val2, reverse=False):
        try:
            v1, v2 = float(val1), float(val2)
            if v1 == v2:
                return ""
            return "✅" if (v1 > v2) != reverse else ""
        except:
            return ""

    stock_list_a = get_symbols(index_a)
    stock_list_b = get_symbols(index_b)

    fundamentals_a = get_fundamentals(stock_a) if stock_a else {}
    fundamentals_b = get_fundamentals(stock_b) if stock_b else {}

    indicators = [
        ("Price", True),
        ("Market Cap", True),
        ("P/E Ratio", True),
        ("EPS", True),
        ("Dividend Yield", False),
        ("Sector", False)
    ]

    # Add just before the return
    raw_symbol_a = stock_a or ""
    raw_symbol_b = stock_b or ""

    return render_template("compare.html",
        user=user,
        index_a=index_a,
        index_b=index_b,
        stock_a=stock_a,
        stock_b=stock_b,
        stock_list_a=stock_list_a,
        stock_list_b=stock_list_b,
        fundamentals_a=fundamentals_a,
        fundamentals_b=fundamentals_b,
        indicators=indicators,
        compare_values=compare_values,
        raw_symbol_a=raw_symbol_a,
        raw_symbol_b=raw_symbol_b
    )




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)



