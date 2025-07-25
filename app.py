import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from Data_Fetch import fetch_stock_data, get_stock_price, get_historical_data, get_news_data
from model import  train, model_predict
import yahoo_fin
from yahoo_fin import news
from sentiment import get_sentiment

st.markdown("""
    <style>
    body {background-color: #0e1117; background-image: url('https://thumbs.dreamstime.com/b/capital-loss-stock-investment-price-falls-monetary-loss-prices-red-angry-investors-sulk-stock-price-plummets-114001949.jpg'); background-size: cover; background-repeat: no-repeat; background-attachment: fixed; color: #ffffff;}
    .stApp {background-color: rgba(14, 17, 23, 0.85);}
    .stSidebar {background-color: #1e2127; color: #ffffff;}
    h1, h2, h3, h4, h5, h6 {color: #ffffff;}
    .stButton>button {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 8px; cursor: pointer;}
    </style>
""", unsafe_allow_html=True)

st.title('Stock Price Prediction App')


st.sidebar.header("Select Company")
selected_company = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOG    )", "AAPL")
time_period = st.sidebar.selectbox("Select Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

if "data" not in st.session_state:
    st.session_state.data = None

if st.sidebar.button("Find Out"):
    #fetch stock data
    stock = get_stock_price(selected_company)

if stock is None:
    st.error("No stock data available for the selected company.")
    st.stop()

    data = get_historical_data(selected_company)

    st.session_state.data = data


    ema100 = data['Close'].ewm(span=100).mean()
    ema200 = data['Close'].ewm(span=200).mean()
    ma100 = data['Close'].rolling(window=100).mean()
    ma200 = data['Close'].rolling(window=200).mean()

    plt.figure(figsize=(15, 8))
    plt.plot(data['Close'], label='Close', color='red')
    plt.plot(ema100, label='EMA 100', color='blue')
    plt.plot(ema200, label='EMA 200', color='green')
    plt.title("Exponential Moving Averages")
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(15, 8))
    plt.plot(data['Close'], label='Close', color='red')
    plt.plot(ma100, label='MA 100', color='cyan')
    plt.plot(ma200, label='MA 200', color='magenta')
    plt.title("Simple Moving Averages")
    plt.legend()
    st.pyplot(plt)



    #sentimnet analysis and news showing
    news_data = get_news_data(selected_company) 
    if news_data is not None:
        st.subheader("Latest News")
        try:
            for i, article in enumerate(news_data[:5]):
                st.write(f"{i+1}. {article['title']}")
                sentiment,polarity = get_sentiment(article['title'])
                st.write(f"🧠 Sentiment: **{sentiment}** (Score: {polarity:.2f})")
        except Exception as e:
            st.error(f"Error displaying news: {e}")
    else:
        st.warning("No news data available for this stock.")




    st.write(f"Current stock price of {selected_company}: ${stock:.2f}")
    #train the model
    train(selected_company)

    #predicting the next day stock price
    if st.session_state.data is not None:
        try:
            prediction = model_predict(st.session_state.data[['Open', 'High', 'Low', 'Volume', 'Close']].values)
            st.write(f"Predicted stock price for {selected_company}: ${prediction[-1]:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.warning("Please click 'Find Out' first to load the data.")



