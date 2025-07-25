from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from Data_Fetch import fetch_stock_data, get_stock_price, get_historical_data
import yfinance as yf
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

 
def train(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")


    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    X = data[['Close', 'Open', 'High', 'Low', 'Volume']]  
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    filename = 'model.pkl'
    pickle.dump(model, open(filename, 'wb'))  


def model_predict(X):
    model = pickle.load(open('model.pkl', 'rb'))
    return model.predict(X)
        