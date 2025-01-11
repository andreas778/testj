import pandas as pd
import datetime
import pytz

def apply_MA(df, ticker=None, window=10, ma_tag=None):
    df[f'{ticker}-{ma_tag}'] = df[ticker].rolling(window=window).mean()
    return df


def apply_strategic_function(df, ticker):
    """
    Applies a trading strategy based on moving averages to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing price data and moving averages.
    - ticker (str): The ticker symbol to identify the columns for the moving averages.

    Returns:
    - pd.DataFrame: The DataFrame with the trading signal column added.
    """
    fast_col = f"{ticker}-MA_fast"  # Column name for the fast moving average
    slow_col = f"{ticker}-MA_slow"  # Column name for the slow moving average
    signal_col = f"{ticker}-signal"  # Column name for the trading signal

    # Initialize the signal column with a default value of 0.5
    df[signal_col] = 0.5

    # Identify where the fast MA crosses above the slow MA (bullish signal)
    cross_above = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
    df.loc[cross_above, signal_col] = 1

    # Identify where the fast MA crosses below the slow MA (bearish signal)
    cross_below = (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))
    df.loc[cross_below, signal_col] = 0

    # Identify where either the fast MA or slow MA is None (incomplete data)
    ma_none = df[fast_col].isnull() | df[slow_col].isnull()
    df.loc[ma_none, signal_col] = None

    return df


def get_opposite_side(side):
    if side == 0:
        return 1
    elif side == 1:
        return 0


class Strategy:
    # Define standard lot sizes for different instruments. The list will be expanded as needed.
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

    number_of_open_trades_in_the_end = 0
    df_news_source, df_news = None, None
    trade_costs = None

    def __init__(self, df):
        """
        Initialize the Strategy class with the provided DataFrame and default values.

        Parameters:
        df (DataFrame): The source DataFrame containing the trading data.
        """
        self.df_source = df  # Original data source
        self.df = None  # Processed DataFrame
        self.deposit = 10000  # Initial deposit
        self.lot_size = 0.01  # Lot size for trading
        self.trade_id = 0  # Trade ID counter

        # Get sorted list of columns and filter out 'time' to get tickers
        columns = df.columns.tolist()
        columns.sort()
        self.tickers = [i for i in columns if i not in ['time', 'Timestamp']]

        self.params = {}  # Parameters for the strategy
        self.trades_by_strategy = 0  # Counter for trades by strategy
        self.trades_not_by_strategy = 0  # Counter for trades not by strategy
        # Dictionaries to track open and closed trades for each ticker
        self.open_trades = {t: {} for t in self.tickers}
        self.closed_trades = {t: [] for t in self.tickers}

    def set_params(self, params=None):
        """
        Set the strategy parameters and process the data accordingly.

        Parameters:
        params (dict): The parameters for the strategy.
        """
        self.params = params  # Set strategy parameters
        self.data_processing()  # Process data with new parameters

    def set_trade_costs(self, trade_costs=None):
        """
        Set the trade costs for different tickers.

        Parameters:
        trade_costs (dict): The trade costs for the strategy.
        """
        self.trade_costs = {}
        for key, values in trade_costs.items():
            self.trade_costs[key] = {}
            for ticker in self.tickers:
                if ticker in values:
                    self.trade_costs[key][ticker] = values[ticker]
                else:
                    self.trade_costs[key][ticker] = values["default"]

    def data_processing(self):
        """
        Process the data by applying moving averages and strategic functions.
        """
        self.df = self.df_source.copy()  # Create a copy of the original DataFrame
        self.df = self.df.reset_index(drop=True)

        # Apply moving averages and strategic functions to the DataFrame
        for ticker in self.tickers:
            for key, value in self.params["MA"][ticker].items():
                self.df = apply_MA(df=self.df, ticker=ticker, window=value, ma_tag=key)
            apply_strategic_function(df=self.df, ticker=ticker)

        self.df["deposit"] = None  # Add a 'deposit' column
        self.df.loc[0, "deposit"] = self.deposit  # Initialize the first row's deposit

    def trading_strategy(self, index):
        """
        Implement the trading strategy for a given index.
        This includes logic to close existing trades and open new trades based on signals.

        Parameters:
        index (int): The current index in the DataFrame for which the strategy is evaluated.

        Returns:
        current_profit (float): The profit made from closing trades at this index.
        """
        current_profit = 0  # Initialize current profit for this index

        # Logic to close trades
        for ticker in self.tickers:
            signal = self.df.loc[index, ticker + "-signal"]  # Get the current signal for the ticker
            if signal != 0.5 and len(
                    self.open_trades[ticker]) > 0:  # If signal is not neutral and there are open trades
                for trade_id, trade in self.open_trades[ticker].copy().items():
                    if pd.isna(signal):  # If the signal is NaN, close the trade due to data issues
                        profit = self.close_trade(
                            id=trade_id,
                            index=index,
                            ticker=ticker,
                            close_price=self.df.loc[index - 1, ticker],  # Use the previous valid price
                            closing_reason="issue_with_data"
                        )
                        current_profit += profit
                        self.trades_not_by_strategy += 1  # Increment the count for trades not by strategy

                    if signal == 1 or signal == 0:  # If the signal indicates to close the trade
                        profit = self.close_trade(
                            id=trade_id,
                            index=index,
                            ticker=ticker,
                            close_price=self.df.loc[index, ticker],  # Use the current price
                            closing_reason="issue_with_data"
                        )
                        current_profit += profit
                        self.trades_by_strategy += 1  # Increment the count for trades by strategy

        # Update the deposit value for the current index
        if index > 0:
            self.df.loc[index, "deposit"] = self.df.loc[index - 1, "deposit"] + current_profit

        # Logic to open trades
        for ticker in self.tickers:
            trade_signal = self.check_open_trade_criterias(index, ticker)  # Check if criteria to open a trade are met

            if trade_signal is not None:
                id = self.open_trade(ticker, index=index, signal=trade_signal)  # Open the trade with the given signal

        return current_profit

    def check_open_trade_criterias(self, index, ticker):
        """
        Check if the criteria to open a trade are met for a given ticker at a given index.

        Parameters:
        index (int): The current index in the DataFrame for which the criteria are checked.
        ticker (str): The ticker symbol for which the criteria are checked.

        Returns:
        trade_signal (int or None): The signal to open a trade (0 or 1), or None if criteria are not met.
        """

        signal = self.df.loc[index, ticker + "-signal"]  # Get the signal for the current ticker
        trade_signal = self.open_trade_criteria_1(signal=signal)  # Check the first criteria

        return trade_signal

    def open_trade_criteria_1(self, signal):
        """
        Determine the trade signal based on the primary signal.

        Parameters:
        signal (int or float): The primary signal for the trade (0, 0.5, or 1).

        Returns:
        trade_signal (int or None): The trade signal (0 or 1) if valid, otherwise None.
        """
        trade_signal = None
        if signal == 1 or signal == 0:
            trade_signal = signal  # Set trade signal if it is 0 or 1
        return trade_signal

    def get_total_profit(self):
        """
        Calculate the total profit from the initial deposit to the current deposit.

        Returns:
        total_profit (float or None): The total profit or None if not calculable.
        """
        try:
            return self.df.iloc[-1]["deposit"] - self.deposit
        except:
            return None

    def close_trade(self, id, index, ticker, close_price, closing_reason):
        """
        Close an existing trade and calculate the profit.

        Parameters:
        id (int): The trade ID.
        index (int): The current index in the DataFrame.
        ticker (str): The ticker symbol for the trade.
        close_price (float): The price at which the trade is closed.
        closing_reason (str): The reason for closing the trade.

        Returns:
        profit (float): The profit made from closing the trade.
        """
        trade = self.open_trades[ticker][id]  # Get the trade details
        trade["price_close"] = close_price  # Set closing price
        trade["index_close"] = index  # Set closing index
        trade["timestamp_close"] = self.df.loc[index, "Timestamp"]
        trade["closing_reason"] = closing_reason  # Set reason for closing trade
        # Calculate profit based on trade side and size
        profit = (trade["price_open"] - trade["price_close"]) * (-1) ** trade["side"] * trade["size"]
        trade["profit"] = profit  # Record the profit
        if self.trade_costs:
            trade = self.calculate_trade_costs(trade)
        self.closed_trades[ticker].append(trade)  # Move trade to closed trades
        self.open_trades[ticker].pop(id)  # Remove trade from open trades
        return trade["profit"]

    def calculate_trade_costs(self, trade):
        """
        Calculate the trade costs including spread, commission, and swap.

        Parameters:
        trade (dict): The trade details.

        Returns:
        trade (dict): The updated trade details with trade costs included.
        """
        ticker = trade['ticker']

        # Calculate spread cost
        spread_pips = self.trade_costs['spread'][ticker]
        spread_cost = spread_pips * self.pips_size[ticker] * trade["size"]

        # Calculate commission cost
        commission_percent = self.trade_costs['comission'][ticker] / 100
        transaction_value = trade['size'] * trade['price_open']
        commission_cost = commission_percent * transaction_value  # Entry and exit included in cost

        # Convert timestamps to datetime objects
        datetime_open = datetime.datetime.fromtimestamp(trade['timestamp_open'], tz=pytz.utc)
        datetime_close = datetime.datetime.fromtimestamp(trade['timestamp_close'], tz=pytz.utc)

        # Calculate the number of midnights between the open and close times
        number_of_midnights = (datetime_close.date() - datetime_open.date()).days

        # Calculate swap cost
        swap_pips = self.trade_costs['swap'][ticker]
        swap_cost = swap_pips * self.pips_size[ticker] * trade["size"] * number_of_midnights

        trade["trade_costs"] = swap_cost + commission_cost + spread_cost
        trade["gross_profit"] = trade["profit"]
        trade["profit"] = trade["profit"] - trade["trade_costs"]

        return trade

    def open_trade(self, ticker, index, signal, stop_loss=None, take_profit=None):
        """
        Open a new trade based on the given signal, with optional limit orders, stop-loss, and take-profit.

        Parameters:
        ticker (str): The ticker symbol for the trade.
        index (int): The current index in the DataFrame.
        signal (int): The signal indicating the trade side (0 for sell, 1 for buy).
        stop_loss (float, optional): The stop-loss price for the trade.
        take_profit (float, optional): The take-profit price for the trade.

        Returns:
        id (int): The trade ID.
        """
        id = self.get_trade_id()  # Generate a new trade ID
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
        self.open_trades[ticker][id] = trade  # Record the new trade

        return id

    def get_trade_id(self):
        """
        Generate a new trade ID.

        Returns:
        trade_id (int): The new trade ID.
        """
        self.trade_id += 1  # Increment the trade ID counter
        return self.trade_id  # Return the new trade ID

    def regular_check_per_tick(self, index):
        """
        Perform regular checks per tick for stop-loss and take-profit conditions.

        Parameters:
        index (int): The current index in the DataFrame.
        """
        for ticker in self.tickers:
            current_price = self.df.loc[index, ticker]
            for id, order in list(self.open_trades[ticker].items()):
                if order["stop_loss"] is not None and \
                        ((order["side"] == 1 and current_price <= order["stop_loss"]) or \
                         (order["side"] == 0 and current_price >= order["stop_loss"])):
                    # Close order due to stop-loss hit
                    self.close_trade(id, index=index, ticker=ticker, close_price=current_price,
                                     closing_reason="Stop-loss")
                elif order["take_profit"] is not None and \
                        ((order["side"] == 1 and current_price >= order["take_profit"]) or \
                         (order["side"] == 0 and current_price <= order["take_profit"])):
                    # Close order due to take-profit hit
                    self.close_trade(id, index=index, ticker=ticker, close_price=current_price,
                                     closing_reason="Take-profit")

    def evaluate(self):
        """
        Evaluate the strategy by iterating over each row in the DataFrame and applying the trading strategy.
        """
        for index, row in self.df.iterrows():  # Iterate over each row in the DataFrame
            # Check and execute limit orders, check stop losses and take profits
            self.regular_check_per_tick(index=index)

            self.trading_strategy(index)  # Apply trading strategy

        self.number_of_open_trades_in_the_end = sum([len(i) for i in self.open_trades.values()])
