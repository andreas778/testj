#!/usr/bin/env python3

import pandas as pd
import pandas_ta as ta  # pip install pandas_ta

# ========= USER PARAMETERS =========
INPUT_CSV_PATH = "/content/testj/system/data/EURCAD-EURGBP-EURUSD-XAUUSD_M1_20190101_20240623.csv"
OUTPUT_CSV_PATH = "/content/testj/system/data/EURCAD-EURGBP-EURUSD-XAUUSD_M1_20190101_20240623_with_indicators.csv"

# Example indicator settings
RSI_LENGTH = 14
CCI_LENGTH = 20
# ===================================

def main():
    # Read data
    df = pd.read_csv(INPUT_CSV_PATH)

    # For each price column, calculate RSI and CCI:
    # We assume the numeric columns (besides Timestamp) are the instruments
    instruments = [col for col in df.columns if col not in ["time", "Timestamp"]]

    for ticker in instruments:
        # RSI
        df[f"{ticker}_RSI"] = ta.rsi(df[ticker], length=RSI_LENGTH)
        # CCI
        df[f"{ticker}_CCI"] = ta.cci(high=df[ticker], low=df[ticker], close=df[ticker],
                                     length=CCI_LENGTH)

    # Save the updated DataFrame
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Adapted data saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
