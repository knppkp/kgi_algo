import pandas as pd
import numpy as np
import os

# Function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, period):
    return data['LastPrice'].ewm(span=period, adjust=False).mean()

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, period=14):
    delta = data['LastPrice'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to make buy/sell decisions and handle cash/stock balances
def execute_transaction(row, portfolio, cash_balance):
    if row['Decision'] == "Buy" and cash_balance >= row['LastPrice'] * 100:
        # Perform buy
        symbol = row['ShareCode']
        cost = row['LastPrice'] * 100
        cash_balance -= cost
        portfolio[symbol] = portfolio.get(symbol, 0) + 100
        return "Buy Executed", cash_balance, portfolio
    elif row['Decision'] == "Sell" and portfolio.get(row['ShareCode'], 0) >= 100:
        # Perform sell
        symbol = row['ShareCode']
        revenue = row['LastPrice'] * 100
        cash_balance += revenue
        portfolio[symbol] -= 100
        if portfolio[symbol] == 0:
            del portfolio[symbol]
        return "Sell Executed", cash_balance, portfolio
    else:
        # Hold or insufficient balance/stocks
        return "Hold", cash_balance, portfolio

# Function to make buy/sell decisions
def make_decision(row):
    if pd.notna(row['predicted_close']):
        if row['predicted_close'] > row['EMA'] and row['RSI'] > 60:
            return "Buy"
        elif row['predicted_close'] < row['EMA'] and row['RSI'] < 20:
            return "Sell"
    return "Hold"

# Load the tick data
tick_file_path = "Daily_Ticks.csv"
tick_data = pd.read_csv(tick_file_path)
tick_data['TradeDateTime'] = pd.to_datetime(tick_data['TradeDateTime'])

# Base path for predictions
base_path = os.path.dirname(os.path.abspath(__file__))

# List of stock symbols
stock_symbols = [
    "ADVANC", "AOT", "AWC", "BBL", "BCP", "BDMS", "BEM", "BGRIM", "BH", "BJC",
    "BTS", "CBG", "CENTEL", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EA", "EGCO",
    "GLOBAL", "GPSC", "GULF", "HMPRO", "INTUCH", "ITC", "IVL", "KBANK", "KTB",
    "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH",
    "SCB", "SCC", "SCGP", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "WHA"
]

# Initialize portfolio and cash balance
portfolio = {}
cash_balance = 1_000_000
transactions = []

# Process each stock symbol
all_decision_data = []

for symbol in stock_symbols:
    print(f"Processing {symbol}...")
    
    # Load the predictions for the symbol
    prediction_file_path = f"{base_path}/rnn/{symbol}_predictions.csv"
    if not os.path.exists(prediction_file_path):
        print(f"Prediction file for {symbol} not found, skipping.")
        continue
    predictions = pd.read_csv(prediction_file_path)
    predictions['interval'] = pd.to_datetime(predictions['interval'])
    
    # Filter tick data for the symbol
    symbol_tick_data = tick_data[tick_data['ShareCode'] == symbol].copy()
    
    # Calculate EMA and RSI
    symbol_tick_data['EMA'] = calculate_ema(symbol_tick_data, period=10)
    symbol_tick_data['RSI'] = calculate_rsi(symbol_tick_data, period=14)
    
    # Match predictions with tick data
    symbol_tick_data['predicted_close'] = np.nan
    for i, pred_row in predictions.iterrows():
        matching_ticks = symbol_tick_data[
            (symbol_tick_data['TradeDateTime'] >= pred_row['interval']) & 
            (symbol_tick_data['TradeDateTime'] < pred_row['interval'] + pd.Timedelta(minutes=5))
        ]
        symbol_tick_data.loc[matching_ticks.index, 'predicted_close'] = pred_row['predicted_close']
    
    # Make decisions
    symbol_tick_data['Decision'] = symbol_tick_data.apply(make_decision, axis=1)
    
    # Execute transactions
    symbol_tick_data['Transaction'] = "Hold"
    for i, row in symbol_tick_data.iterrows():
        transaction_result, cash_balance, portfolio = execute_transaction(row, portfolio, cash_balance)
        symbol_tick_data.at[i, 'Transaction'] = transaction_result
        if transaction_result in ["Buy Executed", "Sell Executed"]:
            transactions.append({
                'DateTime': row['TradeDateTime'],
                'ShareCode': row['ShareCode'],
                'Action': transaction_result.split()[0],
                'Price': row['LastPrice'],
                'Shares': 100,
                'CashBalance': cash_balance
            })
    
    # Save updated tick data for the symbol
    symbol_tick_data.to_csv(f"{base_path}/rnn/{symbol}_with_decisions.csv", index=False)
    all_decision_data.append(symbol_tick_data)

# Combine all data into a single DataFrame
final_decision_data = pd.concat(all_decision_data, ignore_index=True)
final_decision_data.to_csv(f"{base_path}/rnn/all_symbols_with_decisions.csv", index=False)

# Save transactions to a separate file
transactions_df = pd.DataFrame(transactions)
os.makedirs(f"{base_path}/rnnfin", exist_ok=True)
transactions_df.to_csv(f"{base_path}/rnnfin/transaction.csv", index=False)

print("Decisions for all symbols saved to rnn/all_symbols_with_decisions.csv")
print("Buy and Sell transactions saved to rnnfin/transaction.csv")
