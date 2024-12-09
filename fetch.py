import os
import pandas as pd
from datetime import datetime, timezone
from settrade_v2 import Investor
import threading

user_app_id = "M2ze9mceL2lS0RjX"
user_app_secret = "DW8i03GC9d+Yv1bwynPOQsT3/1u4hS5XIo40BsETN5o="
user_broker_id = "SANDBOX"
user_app_code = "SANDBOX"
user_derivatives_account = "kp-D"
user_pin = "000000"

stock_symbols = ["ADVANC", "AOT", "AWC", "BBL", "BCP", "BDMS", "BEM", "BGRIM", "BH", "BJC",
                 "BTS", "CBG", "CENTEL", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EA", "EGCO",
                 "GLOBAL", "GPSC", "GULF", "HMPRO", "INTUCH", "ITC", "IVL", "KBANK", "KTB",
                 "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH",
                 "SCB", "SCC", "SCGP", "TIDLOR", "TISCO", "TLI", "TOP", "TRUE", "TTB", "TU", "WHA"]

# Initialize the API client
investor = Investor(
    app_id=user_app_id,
    app_secret=user_app_secret,
    broker_id=user_broker_id,
    app_code=user_app_code,
    is_auto_queue=False
)
market = investor.MarketData()

# Function to fetch candlestick data
def get_candlestick(symbol, bar_num, interval, data, lock):
    try:
        res = market.get_candlestick(
            symbol=symbol,
            interval=interval,
            limit=bar_num,
            normalized=True,
            start=datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat(),
            end=datetime(2024, 12, 3, tzinfo=timezone.utc).isoformat()
        )

        if res and isinstance(res, dict) and 'time' in res:
            for i in range(len(res['time'])):
                row = {
                    'symbol': symbol,
                    'datetime': datetime.fromtimestamp(res['time'][i], timezone.utc),
                    'open': res['open'][i],
                    'high': res['high'][i],
                    'low': res['low'][i],
                    'close': res['close'][i],
                    'volume': res['volume'][i] if 'volume' in res else None,
                    'value': res['value'][i] if 'value' in res else None,
                }
                with lock:  # Acquire lock before appending to data
                    data.append(row)
        else:
            print(f"Error retrieving data for {symbol}: {res}")
    except Exception as e:
        print(f"An error occurred for {symbol}: {e}")

# Main execution
if __name__ == "__main__":
    data = []
    lock = threading.Lock()  # Create a lock for thread safety

    print("========== Getting 50 Stocks OHLC data. ==============")

    threads = []
    for symbol in stock_symbols:
        thread = threading.Thread(target=get_candlestick, args=(symbol, 100, "5m", data, lock))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("========== All threads have been completed. ==============")

    # Convert to a Pandas DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)
    print("Candlestick data saved.")
