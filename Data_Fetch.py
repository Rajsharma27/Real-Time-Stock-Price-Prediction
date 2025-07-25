import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import yahoo_fin
from yahoo_fin import news

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1]
    return price

def get_historical_data(ticker,period="max"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_stock_data(ticker):
    data = yf.download(tickers=ticker, period="max", interval="1d")
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data[['Open', 'High', 'Low', 'Volume', 'Close']]

def get_news_data(ticker):
    try:
        return news.get_yf_rss(ticker)
    except:
        return None



