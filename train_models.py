import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from train import StockLSTM, SEQUENCE_LENGTH, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
TICKERS = ['PFE', 'JNJ']
PRED_DAYS = 7

def download_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=365)
    df = pdr.DataReader(ticker, 'stooq', start, end)
    return df['Close'].values[::-1]

def train_and_save_model(ticker):
    prices = download_data(ticker)
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    X, y = [], []
    for i in range(len(prices_scaled) - SEQUENCE_LENGTH):
        X.append(prices_scaled[i:i+SEQUENCE_LENGTH])
        y.append(prices_scaled[i+SEQUENCE_LENGTH])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y.squeeze())
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"{MODEL_DIR}/{ticker}_lstm.pt")
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    print(f"Saved model and scaler for {ticker}")

def simulate_sentiment_adjustment(base_preds):
    sentiment_effect = np.random.uniform(-0.1, 0.1, size=len(base_preds))
    return (base_preds * (1 + sentiment_effect)).round(2)

def rolling_predict(model, last_seq, days, scaler):
    model.eval()
    preds_scaled = []
    sequence = last_seq[-SEQUENCE_LENGTH:].copy()

    for _ in range(days):
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(x).item()
        preds_scaled.append(pred_scaled)
        sequence = np.vstack((sequence[1:], [[pred_scaled]]))

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds.tolist(), simulate_sentiment_adjustment(preds)

def save_predictions():
    for ticker in TICKERS:
        model = StockLSTM()
        model.load_state_dict(torch.load(f"{MODEL_DIR}/{ticker}_lstm.pt"))
        scaler = joblib.load(f"{MODEL_DIR}/{ticker}_scaler.pkl")

        prices = download_data(ticker)
        prices_scaled = scaler.transform(prices.reshape(-1, 1))
        last_seq = prices_scaled[-(SEQUENCE_LENGTH + 1):]

        base_preds, sentiment_preds = rolling_predict(model, last_seq, PRED_DAYS, scaler)

        df = pd.DataFrame({
            "date": pd.date_range(start=datetime.now() + timedelta(days=1), periods=PRED_DAYS),
            "base": base_preds,
            "sentiment": sentiment_preds
        })
        df.to_csv(f"{MODEL_DIR}/{ticker}_predictions.csv", index=False)
        print(f"{ticker} base predictions:", base_preds)
        print(f"{ticker} sentiment predictions:", sentiment_preds)

if __name__ == "__main__":
    for ticker in TICKERS:
        train_and_save_model(ticker)
    save_predictions()