import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title {
        font-size: 38px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Tesla Stock Price Prediction using Deep Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive Dashboard with SimpleRNN and LSTM</div>', unsafe_allow_html=True)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.ffill().bfill()
    return df

df = load_data()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["SimpleRNN", "LSTM"]
)

horizon_choice = st.sidebar.selectbox(
    "Choose Prediction Horizon",
    ["1-Day", "5-Day", "10-Day"]
)

show_raw_data = st.sidebar.checkbox("Show Raw Dataset", False)
show_stats = st.sidebar.checkbox("Show Dataset Statistics", False)

# ---------------------------
# Top Summary
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Start Date", str(df["Date"].min().date()))
with col3:
    st.metric("End Date", str(df["Date"].max().date()))
with col4:
    st.metric("Latest Close", f"{df['Close'].iloc[-1]:.2f}")

# ---------------------------
# Dataset Section
# ---------------------------
st.subheader("Tesla Stock Dataset Overview")

if show_raw_data:
    st.dataframe(df.head(20), use_container_width=True)

if show_stats:
    st.dataframe(df.describe(), use_container_width=True)

# ---------------------------
# Interactive Visualization
# ---------------------------
st.subheader("Interactive Closing Price Trend")

date_range = st.slider(
    "Select number of recent days to display",
    min_value=30,
    max_value=min(1000, len(df)),
    value=min(300, len(df))
)

filtered_df = df.tail(date_range)

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(filtered_df["Date"], filtered_df["Close"], label="Close Price")
ax1.set_title("Tesla Closing Price Trend")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price")
ax1.legend()
st.pyplot(fig1)

# ---------------------------
# Moving Average Section
# ---------------------------
st.subheader("Moving Average Analysis")

ma_short = st.selectbox("Select Short Moving Average", [5, 10, 20], index=1)
ma_long = st.selectbox("Select Long Moving Average", [20, 50, 100], index=1)

df["MA_SHORT"] = df["Close"].rolling(ma_short).mean()
df["MA_LONG"] = df["Close"].rolling(ma_long).mean()

ma_df = df.tail(date_range)

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(ma_df["Date"], ma_df["Close"], label="Close Price")
ax2.plot(ma_df["Date"], ma_df["MA_SHORT"], label=f"MA {ma_short}")
ax2.plot(ma_df["Date"], ma_df["MA_LONG"], label=f"MA {ma_long}")
ax2.set_title("Closing Price with Moving Averages")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# ---------------------------
# Prediction Section
# ---------------------------
st.subheader("Model Prediction Results")

LOOKBACK = 60

@st.cache_resource
def load_saved_model(model_choice, horizon_choice):
    model_map = {
        ("SimpleRNN", "1-Day"): "simplernn_1day.keras",
        ("LSTM", "1-Day"): "lstm_1day.keras",
        # Add these later if you save them:
        # ("SimpleRNN", "5-Day"): "simplernn_5day.keras",
        # ("LSTM", "5-Day"): "lstm_5day.keras",
        # ("SimpleRNN", "10-Day"): "simplernn_10day.keras",
        # ("LSTM", "10-Day"): "lstm_10day.keras",
    }
    path = model_map.get((model_choice, horizon_choice))
    if path:
        return load_model(path)
    return None

def create_sequences(dataset, lookback=60, horizon=1):
    X, y = [], []
    for i in range(lookback, len(dataset) - horizon + 1):
        X.append(dataset[i-lookback:i, 0])
        y.append(dataset[i + horizon - 1, 0])
    return np.array(X), np.array(y)

horizon_map = {"1-Day": 1, "5-Day": 5, "10-Day": 10}
horizon = horizon_map[horizon_choice]

data = df[["Close"]].copy()
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

combined = np.concatenate((train_scaled[-LOOKBACK:], test_scaled), axis=0)
X_test, y_test = create_sequences(combined, lookback=LOOKBACK, horizon=horizon)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

model = load_saved_model(model_choice, horizon_choice)

if model is not None:
    preds = model.predict(X_test, verbose=0)
    preds_actual = scaler.inverse_transform(preds)

    mse = mean_squared_error(y_test_actual, preds_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, preds_actual)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("MSE", f"{mse:.4f}")
    with c2:
        st.metric("RMSE", f"{rmse:.4f}")
    with c3:
        st.metric("MAE", f"{mae:.4f}")

    plot_days = st.slider(
        "Select number of prediction points to display",
        min_value=20,
        max_value=min(300, len(y_test_actual)),
        value=min(100, len(y_test_actual))
    )

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(y_test_actual[-plot_days:], label="Actual Price")
    ax3.plot(preds_actual[-plot_days:], label=f"{model_choice} Predicted Price")
    ax3.set_title(f"{model_choice} - {horizon_choice} Prediction")
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Close Price")
    ax3.legend()
    st.pyplot(fig3)

    latest_pred = preds_actual[-1][0]
    latest_actual = y_test_actual[-1][0]

    st.subheader("Latest Prediction Snapshot")
    s1, s2 = st.columns(2)
    with s1:
        st.success(f"Latest Actual Close: {latest_actual:.2f}")
    with s2:
        st.info(f"Latest Predicted Close: {latest_pred:.2f}")

else:
    st.warning(f"Saved model for {model_choice} - {horizon_choice} not found.")
    st.write("Currently supported:")
    st.write("- SimpleRNN 1-Day")
    st.write("- LSTM 1-Day")
    st.write("You can save 5-day and 10-day models later and connect them here.")

# ---------------------------
# Project Summary
# ---------------------------
st.subheader("Project Summary")
st.markdown(f"""
- **Target Variable:** Close Price  
- **Selected Model:** {model_choice}  
- **Prediction Horizon:** {horizon_choice}  
- **Lookback Window:** 60 days  
- **Evaluation Metrics:** MSE, RMSE, MAE  
""")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed as part of Tesla Stock Price Prediction Project using SimpleRNN and LSTM.")