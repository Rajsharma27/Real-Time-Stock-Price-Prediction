import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #0e1117;
    }
    .stSidebar {
        background-color: #1e2127;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Stock Price Prediction App')

st.sidebar.header("Select Company")
selected_company = st.sidebar.selectbox(
    "Select a Company",
    options=["Select a Company", "AAPL", "TSLA", "AMZN", "GOOGL", "MSFT"],  
    index=0  
)

def fetch_stock_data(ticker):
    data = yf.download(tickers=ticker, period="1y", interval="1d")
    if data.empty:
        st.error(f"No data found for ticker '{ticker}'. Please select a valid ticker.")
        return None  
    data.reset_index(inplace=True)
    required_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
    for col in required_columns:
        if col not in data.columns:
            data[col] = np.nan 
    data = data[required_columns]
    return data

def preprocess_data(data, feature_columns, target_column, k):
    data_filtered = data[feature_columns + [target_column]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_filtered)
    data_scaled = pd.DataFrame(data_scaled, columns=feature_columns + [target_column])
    X, y = [], []
    for i in range(k, len(data_scaled)):
        X.append(data_scaled[feature_columns].iloc[i-k:i].values)
        y.append(data_scaled[target_column].iloc[i])
    return np.array(X), np.array(y), scaler

def train_model(X_train, y_train, input_shape, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

if selected_company == "Select a Company":
    st.subheader("Please select a company from the dropdown to proceed.")
else:
    data = fetch_stock_data(selected_company)
    if data is None:
        st.stop()  
    st.subheader(f"Example of Stock Data for {selected_company}")
    st.write(data.head())
    feature_columns = ['Open', 'High', 'Low', 'Volume']
    target_column = 'Close'
    k = 30  

    X, y, scaler = preprocess_data(data, feature_columns, target_column, k)
    X = X.reshape(X.shape[0], X.shape[1], len(feature_columns))
    input_shape = (X.shape[1], X.shape[2])
    model = train_model(X, y, input_shape)

    predicted_scaled = model.predict(X)
    predicted_prices = scaler.inverse_transform(
        np.hstack((X[:, -1, :], predicted_scaled)).reshape(-1, len(feature_columns) + 1)
    )[:, -1]
    actual_prices = scaler.inverse_transform(
        np.hstack((X[:, -1, :], y.reshape(-1, 1))).reshape(-1, len(feature_columns) + 1)
    )[:, -1]
    combined_prices = np.concatenate((actual_prices, predicted_prices))
    x_values = range(len(combined_prices)) 


    
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(actual_prices)), actual_prices, label='Actual Prices', color='blue', linewidth=2)
    plt.plot(range(len(actual_prices) - 1, len(combined_prices)), combined_prices[len(actual_prices) - 1:], label='Predicted Prices', color='orange', linewidth=2)
    plt.title(f"Actual and Predicted Stock Prices for {selected_company}", fontsize=16)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(plt)




    last_k_days = data[feature_columns].values[-k:].reshape(1, k, len(feature_columns))
    next_day_scaled = model.predict(last_k_days)[0, 0]
    placeholder = np.zeros((1, len(feature_columns) + 1))  
    placeholder[0, -1] = next_day_scaled
    next_day_price = scaler.inverse_transform(placeholder)[0, -1]
    ema100 = data['Close'].ewm(span=100, adjust=False).mean()
    ema200 = data['Close'].ewm(span=200, adjust=False).mean()
    ma100 = data['Close'].rolling(window=100).mean()
    ma200 = data['Close'].rolling(window=200).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='red', linewidth=2)
    plt.plot(ema100, label='EMA 100', color='blue', linewidth=2)
    plt.plot(ema200, label='EMA 200', color='green', linewidth=2)
    plt.title(f"Exponential Moving Averages (EMA) for {selected_company}", fontsize=16)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(plt)




    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='red', linewidth=2)
    plt.plot(ma100, label='MA 100', color='cyan', linewidth=2)
    plt.plot(ma200, label='MA 200', color='magenta', linewidth=2)
    plt.title(f"Simple Moving Averages (MA) for {selected_company}", fontsize=16)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    st.pyplot(plt)



    
    st.subheader(f"Predicted Price for the Next Day: ${next_day_price:.2f}")