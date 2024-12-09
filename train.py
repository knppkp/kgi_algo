import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Function to create sequences and labels for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        labels.append(data[i+sequence_length])
    return np.array(sequences), np.array(labels)

# Load and preprocess the data
scaler = MinMaxScaler()
file_path = "data.csv"
df = pd.read_csv(file_path)
df['close_normalized'] = scaler.fit_transform(df[['close']])

# LSTM parameters
sequence_length = 5
num_features = 1
batch_size = 1
epochs = 10

# Build LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, num_features)))
    model.add(Dense(1))  # Predicting the next price
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Predict for a symbol
def predict_for_symbol(symbol, df, model):
    symbol_data = df[df['symbol'] == symbol]
    if len(symbol_data) < sequence_length + 1:
        print(f"Not enough data for symbol {symbol}, skipping.")
        return None
    print(f"Processing {len(symbol_data)} rows for symbol {symbol}.")

    # Prepare sequences and labels
    train_sequences, train_labels = create_sequences(symbol_data['close_normalized'].values, sequence_length)
    train_sequences = train_sequences.reshape((train_sequences.shape[0], sequence_length, num_features))

    # Train the model
    model.fit(train_sequences, train_labels, batch_size=batch_size, epochs=epochs, verbose=0)

    # Get the last sequence from the training data
    last_sequence = train_sequences[-1]

    # Predict the next day's prices (288 intervals)
    predictions_normalized = predict_next_day(model, last_sequence)

    # Inverse transform the predictions to get actual prices
    predicted_prices = scaler.inverse_transform(predictions_normalized.reshape(-1, 1))

    # Create a DataFrame for predictions
    df_predictions = pd.DataFrame(predicted_prices, columns=['predicted_close'])
    df_predictions['interval'] = pd.date_range(
        start=pd.to_datetime('2024-12-04 09:00:00'),  # Specific date and start time for the prediction
        periods=len(predicted_prices),
        freq='5min'
    )

    return df_predictions

# Function to predict prices for the next day (5-minute intervals)
def predict_next_day(model, last_sequence, steps=288):
    predictions = []
    input_sequence = last_sequence

    for _ in range(steps):
        pred = model.predict(input_sequence.reshape((1, sequence_length, num_features)), verbose=0)
        predictions.append(pred[0, 0])  # Store prediction
        input_sequence = np.append(input_sequence[1:], pred, axis=0)  # Shift sequence

    return np.array(predictions)

# Base path for saving results
base_path = os.path.dirname(os.path.abspath(__file__))
os.makedirs(base_path + "/rnn", exist_ok=True)

# List of stock symbols (remove duplicates)
stock_symbols = list(set([
    "GLOBAL", "ADVANC", "AOT", "AWC", "BBL", "BCP", "BDMS", "BEM", "BGRIM", "BH", "BJC",
    "BTS", "CBG", "CENTEL", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EA", "EGCO",
    "GPSC", "GULF", "HMPRO", "INTUCH", "ITC", "IVL", "KBANK", "KTB",
    "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH",
    "SCB", "SCC", "SCGP", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "WHA"
]))

# Group data by symbol
grouped = df.groupby('symbol')

# Initialize model
model = build_model()

# Process predictions for each symbol
all_predictions = []

for symbol in stock_symbols:
    df_predictions = predict_for_symbol(symbol, df, model)
    if df_predictions is None:
        continue
    # Save predictions for each symbol to CSV
    df_predictions.to_csv(f"{base_path}/rnn/{symbol}_predictions.csv", index=False)
    all_predictions.append(df_predictions)

# Combine all predictions into one DataFrame
if all_predictions:
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions.to_csv(f"{base_path}/rnn/all_symbols_predictions.csv", index=False)
    print("All symbols' predictions saved.")
else:
    print("No predictions were generated.")
