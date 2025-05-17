from flask import Flask, render_template, request, redirect, url_for, session
from auth import (
    init_firebase,
    get_user_from_session_cookie,
    verify_password_and_login,
    create_user  # ✅ Must be imported!
)
import os
from tensorflow.keras.models import load_model
model = load_model("PredictModel.keras")
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re

# 🔄 Load .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# 🔐 Initialize Firebase
init_firebase()

@app.before_request
def enforce_session_timeout():
    if 'user' in session:
        now = datetime.utcnow()
        last_active = session.get('last_activity')
        if last_active:
            elapsed = now - datetime.fromisoformat(last_active)
            if elapsed > timedelta(minutes=15):
                session.clear()
                return redirect(url_for("login"))
        session['last_activity'] = now.isoformat()




@app.route("/")
def home():
    import requests
    import os

    user = get_user_from_session_cookie()
    FMP_API_KEY = os.environ.get("FMP_API_KEY")

    def fetch_fmp(endpoint):
        url = f"https://financialmodelingprep.com/api/v3/{endpoint}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url)
            data = response.json()
            return data[:10]  # return top 10 entries
        except:
            return []

    gainers = fetch_fmp("stock_market/gainers")
    losers = fetch_fmp("stock_market/losers")
    actives = fetch_fmp("stock_market/actives")

    return render_template("home.html",
        user=user,
        gainers=gainers,
        losers=losers,
        actives=actives
    )




@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        result = verify_password_and_login(email, password)

        if result == "unverified":
            return render_template("login.html", error="⚠️ Please verify your email before logging in.")
        elif result is True:
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="❌ Invalid email or password")
    
    return render_template("login.html", error=None)



@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    from auth import send_password_reset_email

    success = None
    error = None

    if request.method == "POST":
        email = request.form.get("email")
        if send_password_reset_email(email):
            success = "Reset link sent! Please check your email."
        else:
            error = "Failed to send reset email. Please check your email address."

    return render_template("reset_password.html", success=success, error=error)




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



def is_strong_password(pwd):
    return (
        len(pwd) >= 8 and
        re.search(r"[A-Z]", pwd) and
        re.search(r"[a-z]", pwd) and
        re.search(r"\d", pwd) and
        re.search(r"[!@#$%^&*(),.?\":{}|<>]", pwd)
    )



@app.route("/account", methods=["GET", "POST"])
def account():
    from auth import get_user_from_session_cookie, change_password
    import os
    import requests

    user = get_user_from_session_cookie()
    if not user:
        return redirect(url_for("login"))

    success = None
    error = None

    if request.method == "POST":
        new_password = request.form.get("new_password")

        if not is_strong_password(new_password):
            error = "Password must be at least 8 characters long and include uppercase, lowercase, number, and symbol."
        else:
            # ✅ Get Firebase ID token to authorize password change
            FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY")
            email = user["email"]
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
            resp = requests.post(url, json={
                "email": email,
                "password": request.form.get("current_password", "fake"),
                "returnSecureToken": True
            }).json()

            id_token = resp.get("idToken")

            if not id_token:
                error = "Session expired. Please log in again."
                session.clear()
                return redirect(url_for("login"))

            # ✅ Attempt to change password
            if change_password(id_token, new_password):
                success = "Password updated successfully."
            else:
                error = "Failed to update password. Please try again."

    return render_template("account.html", user=user, success=success, error=error)



@app.route("/chart", methods=["GET", "POST"])
def chart():
    import pandas as pd
    from flask import request, render_template
    user = get_user_from_session_cookie()

    # Load symbol lists from CSVs
    sp500_df = pd.read_csv("sp500_companies.csv")
    ftse100_df = pd.read_csv("FTSE100_Constituents.csv")

    # Create index-to-symbol mapping
    symbol_data = {
        "sp500": sp500_df[["Symbol", "Company", "Sector"]].to_dict(orient="records"),
        "ftse100": ftse100_df[["Symbol", "Company", "Sector"]].to_dict(orient="records")
    }

    selected_index = request.form.get("index", "sp500")
    selected_symbol = request.form.get("symbol")
    full_symbol = None

    if selected_symbol:
        selected_symbol = selected_symbol.strip().upper()
        if selected_symbol.endswith(".L"):
            full_symbol = f"LSE:{selected_symbol}"
        else:
            full_symbol = f"NASDAQ:{selected_symbol}" if selected_index == "sp500" else selected_symbol

    return render_template(
        "chart.html",
        user=user,
        selected_index=selected_index,
        selected_symbol=selected_symbol,
        full_symbol=full_symbol,
        symbol_data=symbol_data
    )



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

    # ✅ Unified watchlist selector
    selected_list = (
        request.form.get("watchlist_name") or
        request.form.get("list_name") or
        request.form.get("selected_list") or
        request.args.get("selected_list")
    )

    index_filter = request.args.get("index_filter") or request.form.get("index_filter") or "all"

    # 🔹 Load stocks by index
    sp500_doc = db.collection("screener_index").document("sp500").get().to_dict() or {}
    ftse_doc = db.collection("screener_index").document("ftse100").get().to_dict() or {}

    sp500_stocks = sp500_doc.get("stocks", [])
    ftse_stocks = ftse_doc.get("stocks", [])

    all_stocks = {
        "sp500": sp500_stocks,
        "ftse100": ftse_stocks,
        "all": sp500_stocks + ftse_stocks
    }.get(index_filter, [])

    # 🔁 POST actions
    if request.method == "POST":
        action = request.form.get("action")
        create_name = request.form.get("create_name", "").strip()
        new_name = request.form.get("new_name", "").strip()
        symbol = request.form.get("symbol", "").strip().upper()

        if action == "create_list" and create_name:
            selected_list = create_name
            doc_ref = watchlist_ref.document(create_name)
            if not doc_ref.get().exists:
                doc_ref.set({"symbols": []})

        elif selected_list:
            doc_ref = watchlist_ref.document(selected_list)
            existing = doc_ref.get().to_dict() or {}
            symbols = existing.get("symbols", [])

            if action == "add" and symbol and symbol not in symbols:
                symbols.append(symbol)
                doc_ref.set({"symbols": symbols}, merge=True)

            elif action == "remove" and symbol in symbols:
                symbols.remove(symbol)
                doc_ref.set({"symbols": symbols}, merge=True)

            elif action == "delete_list":
                doc_ref.delete()
                selected_list = None

            elif action == "rename_list" and new_name:
                old_doc = doc_ref.get().to_dict()
                if old_doc:
                    watchlist_ref.document(new_name).set(old_doc)
                    doc_ref.delete()
                    selected_list = new_name

    # 📥 Load all watchlists
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
        user=user,
        user_watchlists=user_watchlists
    )




@app.route("/predict/<symbol>")
def predict(symbol):
    from firebase_admin import firestore
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from datetime import datetime, timedelta
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io
    import base64
    import os
    import requests

    db = firestore.client()
    user = get_user_from_session_cookie()

    selected_symbol = symbol.upper()
    data = {}
    prediction_plot = None
    forecast_csv = None
    fetching_message = None

    if selected_symbol.endswith(".L"):
        doc = db.collection("stock_cache").document(selected_symbol).get()
        if doc.exists:
            data = doc.to_dict().get("historical_data", {})
    else:
        FMP_API_KEY = os.getenv("FMP_API_KEY")
        fmp_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{selected_symbol}?serietype=line&apikey={FMP_API_KEY}&timeseries=1250"
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
            plt.close(fig)
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')
            prediction_plot = f"data:image/png;base64,{plot_data}"
            forecast_csv = forecast_df.to_csv(index=False)
    else:
        source = "yf" if selected_symbol.endswith(".L") else "fmp"
        try:
            cloud_url = f"https://us-central1-investopred.cloudfunctions.net/updateStockHistory?symbol={selected_symbol}&source={source}"
            requests.get(cloud_url)
            fetching_message = f"⏳ Fetching historical data for {selected_symbol} from {source.upper()}. Please retry in 30–60 seconds."
        except Exception as e:
            fetching_message = f"⚠️ Failed to trigger fetch: {e}"

    return render_template("predict.html",
        user=user,
        symbol=selected_symbol,
        prediction_plot=prediction_plot,
        forecast_csv=forecast_csv,
        fetching_message=fetching_message
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



