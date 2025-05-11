
# 📈 Real-Time Stock Price Prediction

This repository contains a project for predicting stock prices using **LSTM (Long Short-Term Memory)** neural networks and **XGBoost** regression models. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization of predictions.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

The goal of this project is to predict stock prices based on historical data. The project uses **Tesla stock data (`tesladata.csv`)** and implements two models:

- 🧠 **LSTM Neural Network**: For time-series forecasting.
- 📊 **XGBoost Regressor**: As a comparison baseline to LSTM.

The project also performs exploratory data analysis (EDA) and visualizes key indicators such as **Moving Averages (MA)** and **Exponential Moving Averages (EMA)**.

---

## ✨ Features

### 📁 Data Preprocessing
- Handles missing values
- Scales features using `MinMaxScaler`
- Creates time sequences for LSTM input

### 📊 Exploratory Data Analysis (EDA)
- Visualizes stock price trends
- Histograms and boxplots for statistical distribution
- Plots of:
  - 100-day and 200-day **Moving Averages (MA)**
  - 50-day and 100-day **Exponential Moving Averages (EMA)**

### 🧠 Model Training
- Trains an **LSTM** model on historical closing prices
- Trains an **XGBoost** regression model for baseline comparison

### 📈 Visualization
- Actual vs. predicted stock price plots
- Overlay of MAs and EMAs on historical data

### 💾 Model Saving
- Saves trained LSTM model as `Stock_lstm_model.h5`

---

## 🧰 Technologies Used

- **Python** – Core programming language
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **TensorFlow/Keras** – LSTM neural network
- **XGBoost** – Gradient boosting regression
- **Scikit-learn** – Preprocessing and metrics
- **yFinance** – Fetching real-time stock data (optional extension)

---

## 📂 Dataset

The dataset used is `tesladata.csv`, containing historical stock prices of Tesla.

### Key Columns:
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

To run the project:

```bash
python app.py
```

To train the models or visualize results, follow the instructions in the respective Jupyter notebooks or scripts provided in the repository.

---

## 📊 Results

### ✅ LSTM Model:
- Predicts stock prices using temporal sequence data
- Achieves lower **Mean Squared Error (MSE)** than the baseline

### 🔁 XGBoost Model:
- Used as a baseline model
- Slightly higher MSE than LSTM, but fast to train

### 📉 Visualizations:
- **Actual vs. Predicted Prices** – Compare true vs. forecasted values
- **Moving Averages (MA)** – 100-day and 200-day trendlines
- **Exponential Moving Averages (EMA)** – Smoother trend analysis

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repo and submit a pull request.

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---
