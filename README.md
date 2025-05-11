# Real-Time-Stock-Price-Prediction

Stock Price Prediction
This repository contains a project for predicting stock prices using LSTM (Long Short-Term Memory) neural networks and XGBoost regression models. The project includes data preprocessing, exploratory data analysis (EDA), model training, and visualization of predictions.

Table of Contents
Overview
Features
Technologies Used
Dataset
Installation
Usage
Results
Contributing
License
Overview
The goal of this project is to predict stock prices based on historical data. The project uses Tesla stock data (tesladata.csv) and implements two models:

LSTM Neural Network: For time-series forecasting.
XGBoost Regressor: For comparison with the LSTM model.
The project also includes exploratory data analysis (EDA) and visualization of moving averages (MA) and exponential moving averages (EMA).

Features
Data Preprocessing:

Handles missing values.
Scales data using MinMaxScaler.
Creates sequences for LSTM input.
Exploratory Data Analysis (EDA):

Visualizes stock price trends.
Plots histograms and boxplots for key features.
Calculates and plots moving averages (MA) and exponential moving averages (EMA).
Model Training:

Trains an LSTM model for stock price prediction.
Trains an XGBoost regressor for comparison.
Visualization:

Plots actual vs. predicted stock prices.
Displays moving averages and exponential moving averages.
Model Saving:

Saves the trained LSTM model as Stock_lstm_model.h5.
Technologies Used
Python: Programming language.
Pandas: For data manipulation.
NumPy: For numerical computations.
Matplotlib: For data visualization.
TensorFlow/Keras: For building and training the LSTM model.
XGBoost: For regression modeling.
scikit-learn: For preprocessing and evaluation metrics.
yFinance: For fetching stock data.

Dataset
The dataset used in this project is tesladata.csv, which contains historical stock prices for Tesla. The key columns include:

Date
Open
High
Low
Close
Volume

Installation
1.Clone the repository:
git clone https://github.com/<your-username>/Stock-Price-Prediction.git
cd Stock-Price-Prediction
2.Install the required dependencies:
pip install -r requirements.txt


Results
LSTM Model:
The LSTM model predicts stock prices based on historical data.
The model achieves a low mean squared error (MSE) on the test set.
XGBoost Model:
The XGBoost regressor is used as a baseline for comparison.
The MSE of the XGBoost model is slightly higher than the LSTM model.
Visualizations:
Actual vs. Predicted Prices: Shows how well the models predict stock prices.
Moving Averages (MA): Highlights trends using 100-day and 200-day moving averages.
Exponential Moving Averages (EMA): Provides a smoothed view of stock price trends.
