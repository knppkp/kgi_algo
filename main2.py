import numpy as np
import pandas as pd
import os
from datetime import datetime

# Define paths for reading and writing files
file_path = '~/Desktop/Daily_Ticks.csv'
previous_file_path = '~/Desktop/Previous/Result.csv'

output_dir = '~/Desktop/competition_api/Result'
os.makedirs(f'{output_dir}/portfolio', exist_ok=True)
os.makedirs(f'{output_dir}/statement', exist_ok=True)
os.makedirs(f'{output_dir}/summary', exist_ok=True)

# Read the input data
data = pd.read_csv(file_path)
previous_data = pd.read_csv(previous_file_path)

# Preprocess the data
data['TradeDateTime'] = pd.to_datetime(data['TradeDateTime'])  # Convert to datetime
data.sort_values('TradeDateTime', inplace=True)  # Sort by datetime

# Define technical indicators
def calculate_moving_average(df, window=5):
    df[f'MA_{window}'] = df['LastPrice'].rolling(window=window).mean()
    return df

def calculate_volume_signal(df, threshold=1000):
    df['VolumeSignal'] = np.where(df['Volume'] > threshold, 'Buy', 'Hold')
    return df

# Apply technical indicators
data = calculate_moving_average(data, window=5)
data = calculate_volume_signal(data, threshold=1000)

data.to_csv('processed_data.csv', index=False)

# Portfolio settings
initial_cash = 10_000_000
portfolio = {
    "cash": initial_cash,
    "stocks": {},  # Dictionary to store stock holdings
}

# Function to calculate portfolio value
def calculate_portfolio_value(last_prices):
    stock_value = sum(portfolio["stocks"].get(code, 0) * price for code, price in last_prices.items())
    total_value = portfolio["cash"] + stock_value
    return portfolio["cash"], stock_value, total_value

# Function to execute a trade
def execute_trade(stock_code, price, volume, trade_type):
    global portfolio
    
    # Calculate trade value
    trade_value = price * volume
    
    if trade_type == "Buy":
        if portfolio["cash"] >= trade_value:
            portfolio["cash"] -= trade_value
            portfolio["stocks"][stock_code] = portfolio["stocks"].get(stock_code, 0) + volume
            print(f"Bought {volume} of {stock_code} at {price} THB")
        else:
            print(f"Not enough cash to buy {volume} of {stock_code} at {price} THB")
    elif trade_type == "Sell":
        if portfolio["stocks"].get(stock_code, 0) >= volume:
            portfolio["cash"] += trade_value
            portfolio["stocks"][stock_code] -= volume
            print(f"Sold {volume} of {stock_code} at {price} THB")
        else:
            print(f"Not enough stocks to sell {volume} of {stock_code} at {price} THB")

last_prices = {}

for _, row in data.iterrows():
    stock_code = row["ShareCode"]
    price = row["LastPrice"]
    volume = row["Volume"]
    flag = row["Flag"]
    
    if flag not in ["Buy", "Sell"]:
        continue

    if row['VolumeSignal'] == 'Buy' and row['Flag'] == 'Sell':
        cash = portfolio["cash"]
        shares_to_buy = cash // price
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            if row["Volume"] < shares_to_buy:
                shares_to_buy = row["Volume"]
        execute_trade(stock_code, price, shares_to_buy, "Buy")
    
    elif row['VolumeSignal'] == 'Hold' and row['Flag'] == 'Buy':
        if stock_code in last_prices:
            shares_to_sell = portfolio["stocks"].get(stock_code, 0)
            if shares_to_sell > 0:
                execute_trade(stock_code, price, shares_to_sell, "Sell")
    
    # Update last prices
    last_prices[stock_code] = price

# Final portfolio summary
cash_balance, stock_value, total_value = calculate_portfolio_value(last_prices)
print("\nEnd of Day Portfolio Summary:")
print(f"Cash Balance: {cash_balance:.2f} THB")
print(f"Stock Holdings Value: {stock_value:.2f} THB")
print(f"Total Portfolio Value: {total_value:.2f} THB")

start_value = initial_cash
end_value = total_value
return_percentage = (end_value - start_value) / start_value * 100
print(f"\n% Return: {return_percentage:.2f}%")

# Prepare the Portfolio Table for export
portfolio_data = []
for stock_code, volume in portfolio["stocks"].items():
    price = last_prices.get(stock_code, 0)
    amount_cost = volume * price
    market_value = volume * price
    unrealized_pl = market_value - amount_cost
    unrealized_pl_pct = unrealized_pl / amount_cost * 100 if amount_cost > 0 else 0
    
    portfolio_data.append({
        'Table Name': 'Portfolio',
        'File Name': '013_KGI_portfolio.csv',
        'Stock Name': stock_code,
        'Start Vol': volume,
        'Actual Vol': volume,
        'Avg Cost': price,
        'Market Price': price,
        'Amount Cost': amount_cost,
        'Market Value': market_value,
        'Unrealized P/L': unrealized_pl,
        '%Unrealized P/L': unrealized_pl_pct,
        'Realized P/L': 0  # Assuming no realized P/L for simplicity
    })

portfolio_df = pd.DataFrame(portfolio_data)
portfolio_df.to_csv(f'{output_dir}/portfolio/017_KGI_portfolio.csv', index=False)

# Prepare the Statement Table for export
statement_data = []
for _, row in data.iterrows():
    statement_data.append({
        'Table Name': 'Statement',
        'File Name': '013_KGI_statement.csv',
        'Stock Name': row['ShareCode'],
        'Date': row['TradeDateTime'].strftime('%Y-%m-%d'),
        'Time': row['TradeDateTime'].strftime('%H:%M:%S'),
        'Side': row['Flag'],
        'Volume': row['Volume'],
        'Actual Vol': row['Volume'],
        'Price': row['LastPrice'],
        'Amount Cost': row['Volume'] * row['LastPrice'],
        'End Line Available': portfolio["cash"],
        'Portfolio Value': total_value,
        'NAV': total_value / initial_cash if initial_cash > 0 else 0
    })

statement_df = pd.DataFrame(statement_data)
statement_df.to_csv(f'{output_dir}/statement/017_KGI_statement.csv', index=False)

# Prepare the Summary Table for export
summary_data = [{
    'Table Name': 'Summary',
    'File Name': '013_KGI_summary.csv',
    'NAV': total_value / initial_cash if initial_cash > 0 else 0,
    'End Line Available': portfolio["cash"],
    'Start Line Available': initial_cash,
    'Number of Wins': 0,
    'Number of Matched Trades': 0,
    'Number of Transactions': len(data),
    'Sum of Unrealized P/L': sum([row['Volume'] * row['LastPrice'] - row['Volume'] * last_prices.get(row['ShareCode'], 0) for _, row in data.iterrows()]),
    'Sum of %Unrealized P/L': 0,
    'Sum of Realized P/L': 0,
    'Maximum Value': max([row['LastPrice'] for _, row in data.iterrows()]),
    'Minimum Value': min([row['LastPrice'] for _, row in data.iterrows()]),
    'Win Rate': 0,
    'Calmar Ratio': 0,
    'Relative Drawdown': 0,
    'Maximum Drawdown': 0,  
    '%Return': return_percentage
}]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{output_dir}/summary/017_KGI_summary.csv', index=False)

print("CSV files exported successfully.")
