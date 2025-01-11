#!/usr/bin/env python3

import os
import pandas as pd
import datetime
import pytz

# ========= USER PARAMETERS =========
DATA_FILE = "data/EURCAD-EURGBP-EURUSD-XAUUSD_M1_20190101_20240623_with_indicators.csv"
TRADE_LOG_FILE = "trade_logs/trade_log.csv"

STARTING_BALANCE = 10_000
LOT_SIZE = 0.01  # fraction of a standard lot
TRADE_COSTS = {
    "spread": {"default": 0.2, "EURUSD": 0.1},
    "comission": {"default": 0.01},  # in percent
    "swap": {"default": 1, "EURUSD": 0.5},
}

# RSI & CCI thresholds (example):
# - RSI > 70 + CCI > 100 => bullish signal
# - RSI < 30 + CCI < -100 => bearish signal
# Everything else => 0.5 (neutral)
RSI_BULLISH = 70
RSI_BEARISH = 30
CCI_BULLISH = 100
CCI_BEARISH = -100

# ===================================

def get_signal(row, ticker):
    """
    Determine the trading signal (0=sell, 1=buy, 0.5=neutral) 
    based on RSI & CCI thresholds.
    """

    rsi_col = f"{ticker}_RSI"
    cci_col = f"{ticker}_CCI"

    if pd.isna(row[rsi_col]) or pd.isna(row[cci_col]):
        return None  # incomplete data

    rsi_val = row[rsi_col]
    cci_val = row[cci_col]

    # Bullish
    if rsi_val > RSI_BULLISH and cci_val > CCI_BULLISH:
        return 1
    # Bearish
    elif rsi_val < RSI_BEARISH and cci_val < CCI_BEARISH:
        return 0
    else:
        # Neutral
        return 0.5


def minutes_between(timestamp_open, timestamp_close):
    """
    Return the difference in minutes between two UNIX timestamps (UTC).
    """
    dt_open = datetime.datetime.fromtimestamp(timestamp_open, tz=pytz.utc)
    dt_close = datetime.datetime.fromtimestamp(timestamp_close, tz=pytz.utc)
    diff = dt_close - dt_open
    return diff.total_seconds() / 60.0


class Strategy:
    # Define standard lot sizes and pips sizes for instruments
    pips_size = {
        "EURCAD": 0.0001,
        "EURGBP": 0.0001,
        "EURUSD": 0.0001,
        "XAUUSD": 0.01
    }
    standard_lot = {
        "EURCAD": 100_000,
        "EURGBP": 100_000,
        "EURUSD": 100_000,
        "XAUUSD": 100
    }

    df_news_source, df_news = None, None

    def __init__(self, df, starting_balance=STARTING_BALANCE, trade_costs=TRADE_COSTS):
        self.df_source = df.copy()
        self.df = None
        self.deposit = starting_balance
        self.lot_size = LOT_SIZE
        self.trade_costs = trade_costs
        self.trade_id = 0

        # Identify tickers
        columns = df.columns.tolist()
        columns.sort()
        self.tickers = [c for c in columns if c not in ["time", "Timestamp"]
                        and not c.endswith("_RSI") and not c.endswith("_CCI")]

        # Counters and data structures
        self.trades_by_strategy = 0
        self.trades_not_by_strategy = 0
        self.open_trades = {t: {} for t in self.tickers}
        self.closed_trades = {t: [] for t in self.tickers}
        self.number_of_open_trades_in_the_end = 0

        # Prepare data
        self.data_processing()

    def data_processing(self):
        self.df = self.df_source.reset_index(drop=True)
        self.df["deposit"] = None
        self.df.loc[0, "deposit"] = self.deposit

    def evaluate(self):
        for index, row in self.df.iterrows():
            self.regular_check_per_tick(index)
            self.trading_strategy(index)
        self.number_of_open_trades_in_the_end = sum([len(v) for v in self.open_trades.values()])
        self.save_trade_log()

    def trading_strategy(self, index):
        current_profit = 0

        # Step 1: Possibly close trades if signals differ from existing positions
        for ticker in self.tickers:
            signal = get_signal(self.df.loc[index], ticker)  # current RSI+CCI-based signal

            # If we have open trades for this ticker
            if len(self.open_trades[ticker]) > 0:
                for trade_id, trade in list(self.open_trades[ticker].items()):
                    # If signal is None => close due to data issues
                    if signal is None:
                        profit = self.close_trade(
                            id=trade_id,
                            index=index,
                            ticker=ticker,
                            close_price=self.df.loc[index-1, ticker] if index > 0 else self.df.loc[index, ticker],
                            closing_reason="issue_with_data"
                        )
                        current_profit += profit
                        self.trades_not_by_strategy += 1
                    else:
                        # If we have a bullish signal (1) but the trade side is 0 => close the sell
                        if signal == 1 and trade["side"] == 0:
                            profit = self.close_trade(
                                id=trade_id,
                                index=index,
                                ticker=ticker,
                                close_price=self.df.loc[index, ticker],
                                closing_reason="signal_opposite"
                            )
                            current_profit += profit
                            self.trades_by_strategy += 1

                        # If we have a bearish signal (0) but the trade side is 1 => close the buy
                        elif signal == 0 and trade["side"] == 1:
                            profit = self.close_trade(
                                id=trade_id,
                                index=index,
                                ticker=ticker,
                                close_price=self.df.loc[index, ticker],
                                closing_reason="signal_opposite"
                            )
                            current_profit += profit
                            self.trades_by_strategy += 1

                        # If we have a same-direction signal (1 -> buy, 0 -> sell) 
                        # while already open in that direction -> do nothing (keep it)
                        # If signal == 0.5 -> neutral, we do NOT necessarily close 
                        # if we want to keep the position open. But you can adapt.
                        # Example: keep open if the same direction, do nothing.
                        # So no action in that scenario.

        # Step 2: Update deposit
        if index > 0:
            self.df.loc[index, "deposit"] = self.df.loc[index - 1, "deposit"] + current_profit

        # Step 3: Possibly open new trades
        for ticker in self.tickers:
            signal = get_signal(self.df.loc[index], ticker)
            if signal is None:
                continue  # skip if incomplete data
            if signal in [0, 1]:
                # Check if we already have an open trade in the same direction
                same_dir_open = any(t["side"] == signal for t in self.open_trades[ticker].values())
                if not same_dir_open:
                    self.open_trade(ticker, index, signal)

    def regular_check_per_tick(self, index):
        # Stop-loss / take-profit checks if used
        # For simplicity, not implementing SL/TP in this example
        pass

    def open_trade(self, ticker, index, signal, stop_loss=None, take_profit=None):
        id = self.get_trade_id()
        trade = {
            "id": id,
            "ticker": ticker,
            "index_open": index,
            "timestamp_open": self.df.loc[index, "Timestamp"],
            "price_open": self.df.loc[index, ticker],
            "size": self.lot_size * self.standard_lot[ticker],
            "side": signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        self.open_trades[ticker][id] = trade

    def close_trade(self, id, index, ticker, close_price, closing_reason):
        trade = self.open_trades[ticker][id]
        trade["price_close"] = close_price
        trade["index_close"] = index
        trade["timestamp_close"] = self.df.loc[index, "Timestamp"]
        trade["closing_reason"] = closing_reason

        # Profit calc
        profit = (trade["price_open"] - trade["price_close"]) * ((-1) ** trade["side"]) * trade["size"]
        trade["profit"] = profit

        # Apply trade costs
        trade = self.calculate_trade_costs(trade)

        # Move to closed trades
        self.closed_trades[ticker].append(trade)
        self.open_trades[ticker].pop(id)
        return trade["profit"]

    def calculate_trade_costs(self, trade):
        #ticker = trade["ticker"]
        ticker = list(self.trade_costs["spread"].keys())[1]

        # Spread
        spread_pips = self.trade_costs["spread"][ticker]
        spread_cost = spread_pips * self.pips_size[ticker] * trade["size"]

        # Commission
        commission_percent = self.trade_costs["comission"].get(ticker, self.trade_costs["comission"].get('default', 0.01)) / 100.0
        transaction_value = trade["size"] * trade["price_open"]
        commission_cost = commission_percent * transaction_value  # simplistic

        # Swap (per day)
        datetime_open = datetime.datetime.fromtimestamp(trade["timestamp_open"], tz=pytz.utc)
        datetime_close = datetime.datetime.fromtimestamp(trade["timestamp_close"], tz=pytz.utc)
        number_of_midnights = (datetime_close.date() - datetime_open.date()).days

        swap_pips = self.trade_costs["swap"][ticker]
        swap_cost = swap_pips * self.pips_size[ticker] * trade["size"] * number_of_midnights

        trade["trade_costs"] = spread_cost + commission_cost + swap_cost
        trade["gross_profit"] = trade["profit"]
        trade["profit"] -= trade["trade_costs"]

        return trade

    def get_trade_id(self):
        self.trade_id += 1
        return self.trade_id

    def save_trade_log(self):
        """
        Save or print the trade log with:
        open time, close time, open price, close price, outcome($), 
        duration(minutes), instrument, etc.
        """
        all_trades = []
        for ticker, trades in self.closed_trades.items():
            for t in trades:
                row = {
                    "ticker": t["ticker"],
                    "open_time": t["timestamp_open"],
                    "close_time": t["timestamp_close"],
                    "open_price": t["price_open"],
                    "close_price": t["price_close"],
                    "outcome($)": round(t["profit"], 2),
                    "duration(minutes)": round(
                        minutes_between(t["timestamp_open"], t["timestamp_close"]), 2
                    ),
                    "side": "BUY" if t["side"] == 1 else "SELL",
                    "closing_reason": t["closing_reason"]
                }
                all_trades.append(row)

        df_log = pd.DataFrame(all_trades)
        if not os.path.exists(os.path.dirname(TRADE_LOG_FILE)):
            os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
        df_log.to_csv(TRADE_LOG_FILE, index=False)
        print(f"Trade log saved to: {TRADE_LOG_FILE}")


def main():
    # Load data with RSI & CCI columns
    df = pd.read_csv(DATA_FILE)

    # Initialize & run the strategy
    strategy = Strategy(df=df, starting_balance=STARTING_BALANCE, trade_costs=TRADE_COSTS)
    strategy.evaluate()

    # Print final deposit
    final_deposit = strategy.df.loc[strategy.df.index[-1], "deposit"]
    print(f"Final deposit: {final_deposit:.2f}")

if __name__ == "__main__":
    main()
