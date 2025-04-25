import numpy as np
import joblib
import os
import torch
import torch.nn as nn

SEQUENCE_LENGTH = 7
MODEL_DIR = "model"

# Define the LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, lstm_size=200, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Dummy price predictor using loaded model
class PricePredictor:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def predict_next_day(self, sequence):
        if isinstance(self.model, torch.nn.Module):
            # Handle PyTorch model
            sequence = np.array(sequence[-SEQUENCE_LENGTH:])
            if self.scaler:
                sequence = self.scaler.transform(sequence.reshape(-1, 1)).reshape(1, -1)
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(sequence).reshape(1, SEQUENCE_LENGTH, 1)
                prediction = self.model(sequence_tensor)
                if self.scaler:
                    prediction = self.scaler.inverse_transform(prediction.numpy())[0][0]
                else:
                    prediction = prediction.item()
            return prediction
        else:
            # Handle other model types
            sequence = np.array(sequence[-SEQUENCE_LENGTH:]).reshape(1, -1)
            return self.model.predict(sequence)[0]

def load_models():
    models = {}
    sentiment_clf = None
    
    # Suppress scikit-learn warnings about version mismatches
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
    
    try:
        # Load stock prediction models
        for stock in ["pfe", "jnj"]:
            model_path = os.path.join(MODEL_DIR, f"{stock.upper()}_lstm.pt")
            scaler_path = os.path.join(MODEL_DIR, f"{stock.upper()}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    # Initialize the model architecture with correct dimensions
                    model = LSTMModel(input_size=1, hidden_size=50, lstm_size=200, num_layers=2)
                    
                    # Load the state dictionary
                    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                    if isinstance(state_dict, dict):
                        if 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        model.load_state_dict(state_dict)
                    
                    model.eval()  # Set to evaluation mode
                    
                    # Load scaler
                    with open(scaler_path, 'rb') as f:
                        scaler = joblib.load(f)
                    
                    models[stock] = (model, scaler)
                    print(f"Successfully loaded {stock.upper()} model and scaler")
                except Exception as e:
                    print(f"Error loading {stock} model: {str(e)}")
                    models[stock] = None
            else:
                print(f"Model files not found for {stock}")
                models[stock] = None
        
        # Load sentiment classifier
        sentiment_path = os.path.join(MODEL_DIR, "sentiment_classifier.pkl")
        if os.path.exists(sentiment_path):
            try:
                with open(sentiment_path, 'rb') as f:
                    sentiment_clf = joblib.load(f)
                print("Successfully loaded sentiment classifier")
            except Exception as e:
                print(f"Error loading sentiment classifier: {str(e)}")
                sentiment_clf = None
    
    except Exception as e:
        print(f"General error loading models: {str(e)}")
    
    return models, sentiment_clf

def get_historical_prices(ticker):
    from pandas_datareader import data as pdr
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=30)
    df = pdr.DataReader(ticker, "stooq", start, end)
    return df["Close"].values[::-1]

def get_enhanced_predictions():
    models, sentiment_clf = load_models()
    results = {}

    for stock in ["pfe", "jnj"]:
        try:
            history = get_historical_prices(stock.upper())
            
            # Handle case where model is not loaded
            if stock not in models or models[stock] is None:
                print(f"Warning: Model for {stock} not found, using fallback predictions")
                # Fallback to simple prediction
                last_price = history[-1] if len(history) > 0 else 100.0  # Default if no history
                future = [last_price + np.random.normal(0, 0.5) for _ in range(7)]
                results[stock] = {
                    "base_predictions": [last_price] * 7,
                    "predictions": future
                }
                continue

            model, scaler = models[stock]
            predictor = PricePredictor(model, scaler)
            future = []

            seq = list(history[-SEQUENCE_LENGTH:])
            for _ in range(7):
                pred = predictor.predict_next_day(seq)
                future.append(float(pred))  # Convert to float to ensure JSON serializable
                seq = np.append(seq[1:], [pred], axis=0)

            results[stock] = {
                "base_predictions": [float(seq[-1])] * 7,  # Convert to float
                "predictions": future
            }
        except Exception as e:
            print(f"Error generating predictions for {stock}: {str(e)}")
            # Fallback values if something goes wrong
            results[stock] = {
                "base_predictions": [100.0] * 7,
                "predictions": [100.0 + np.random.normal(0, 0.5) for _ in range(7)]
            }

    return results

def get_all_sentiment_classes():
    _, sentiment_clf = load_models()
    classes = {}
    for stock in ["pfe", "jnj"]:
        if sentiment_clf:
            dummy_features = np.random.rand(1, 7)
            label = sentiment_clf.predict(dummy_features)[0]
            label_name = ["Negative", "Neutral", "Positive"][label % 3]
        else:
            label_name = "Neutral"
        classes[stock] = label_name
    return classes