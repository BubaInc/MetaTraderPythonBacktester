import MetaTrader5 as mt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from datetime import datetime
import pytz
from tqdm import tqdm
import math


class Timeframe:
    def __init__(self, timeframe, freq) -> None:
        self.timeframe = timeframe
        self.freq = freq


class Timeframes:
    M1 = Timeframe(mt.TIMEFRAME_M1, '1min')
    M2 = Timeframe(mt.TIMEFRAME_M2, '2min')
    M3 = Timeframe(mt.TIMEFRAME_M3, '3min')
    M4 = Timeframe(mt.TIMEFRAME_M4, '4min')
    M5 = Timeframe(mt.TIMEFRAME_M5, '5min')
    M6 = Timeframe(mt.TIMEFRAME_M6, '6min')
    M10 = Timeframe(mt.TIMEFRAME_M10, '10min')
    M12 = Timeframe(mt.TIMEFRAME_M12, '12min')
    M15 = Timeframe(mt.TIMEFRAME_M15, '15min')
    M20 = Timeframe(mt.TIMEFRAME_M20, '20min')
    M30 = Timeframe(mt.TIMEFRAME_M30, '30min')
    H1 = Timeframe(mt.TIMEFRAME_H1, '1H')
    H2 = Timeframe(mt.TIMEFRAME_H2, '2H')
    H3 = Timeframe(mt.TIMEFRAME_H3, '3H')
    H4 = Timeframe(mt.TIMEFRAME_H4, '4H')
    H6 = Timeframe(mt.TIMEFRAME_H6, '6H')
    H8 = Timeframe(mt.TIMEFRAME_H8, '8H')
    H12 = Timeframe(mt.TIMEFRAME_H12, '12H')
    D1 = Timeframe(mt.TIMEFRAME_D1, '1D')
    W1 = Timeframe(mt.TIMEFRAME_W1, '1W')
    MN1 = Timeframe(mt.TIMEFRAME_MN1, '1M')


class Strategy:
    def __init__(
            self,
            symbols_to_test: list[str] = [], 
            timeframe: Timeframe = Timeframes.D1, 
            date_from: datetime = datetime(2020, 1, 1),
            date_to: datetime = datetime(2023, 4, 28),
            percentage_empty_frames=0.5) -> None:

        # initialize MetaTrader5
        env_values = dotenv_values('.env')
        if not mt.initialize(login=int(env_values['MT_LOGIN']), password=env_values['MT_PASSWORD'], server=env_values['MT_SERVER']):
            print('MetaTrader5 did not open! Maybe check your credentials?')
            quit()

        # maps symbol strings to DataFrames
        dataframes = {}

        # use all available symbols when no symbols have been specified
        if len(symbols_to_test) == 0:
            symbols_to_test = [symbol.name for symbol in mt.symbols_get()]

        # create date range with all the indices
        self.date_range = pd.date_range(date_from, date_to, freq=timeframe.freq)
        self.number_of_bars = len(self.date_range)

        # fetch bar data
        number_of_empty_frames = 0
        for symbol in tqdm(symbols_to_test):
            df = pd.DataFrame(mt.copy_rates_range(
                symbol, timeframe.timeframe, date_from, date_to))
            if len(df) == 0:
                number_of_empty_frames += 1
            else:
                df['point'] = mt.symbol_info(symbol).point # store the point value in a seperate column
                df['time'] = pd.to_datetime(df['time'], unit='s') # use the datetime as the index
                df.set_index(df['time'], inplace=True)
                df.drop('time', axis=1,inplace=True)
                df = df.reindex(self.date_range) # add NaN rows for each missing period
                df['can_trade'] = df.isna().all(axis=1)
                df['can_trade'] = ~df['can_trade'] # add can_trade information
                dataframes[symbol] = df # store the dataframe in the dictionary
            if number_of_empty_frames > percentage_empty_frames * len(symbols_to_test): # alert when too many dataframes (half of them by default) are empty
                print('More than 50% of the symbols have no bar data. Maybe you requested too many of them? In that case, try a larger timeframe or a smaller range. If that behavior is intended, change the percentage_empty_frames parameter')
                quit()

        # Create MultiIndex DataFrame for the data
        self.data = pd.concat(dataframes.values(), keys=dataframes.keys())

        # DataFrame that stores the open positions
        self.positions = pd.DataFrame(
            columns=['symbol', 'type', 'volume', 'sl', 'tp', 'price', 'opened_at'])
        self.position_count = 0

        # track statistics
        self.capital_record = pd.DataFrame({'time': [], 'capital': []})
        self.trade_record = pd.DataFrame(
            columns=['symbol', 'type', 'volume', 'pnl', 'opened_at', 'closed_at'])

    def on_bar(self, data, current_bars):
        pass

    def open_position(self, type, symbol, volume, sl=0.0, tp=0.0):
        if type == 'BUY' and not symbol in self.positions.index:
            self.positions.loc[self.position_count] = [symbol, type, volume,
                                          sl, tp, self.current_prices.loc[symbol]['ask'], self.current_date]
        elif type == 'SELL' and not symbol in self.positions.index:
            self.positions.loc[self.position_count] = [symbol, type, volume,
                                          sl, tp, self.current_prices.loc[symbol]['bid'], self.current_date]
        self.position_count += 1

    def close_position(self, id):
        position = self.positions.loc[id]
        if position['type'] == 'BUY':
            pnl = (self.current_prices.loc[position['symbol']]['bid'] -
                   position['price']) * position['volume']
            self.capital += pnl
            self.trade_record.loc[len(self.trade_record)] = [
                position['symbol'], 'BUY', position['volume'], pnl, position['opened_at'], self.current_date]
            self.positions = self.positions.drop(id)
        elif position['type'] == 'SELL':
            pnl = (position['price'] - self.current_prices.loc[position['symbol']]
                   ['ask']) * position['volume']
            self.capital += pnl
            self.trade_record.loc[len(self.trade_record)] = [
                position['symbol'], 'SELL', position['volume'], pnl, position['opened_at'], self.current_date]
            self.positions = self.positions.drop(id)
    
    def is_position_open(self, symbol, type):
        for i in range(len(self.positions)):
            position = self.positions.iloc[i]
            if position['symbol'] == symbol and position['type'] == type:
                return True
        return False

    def run(self, window=3, start_capital=1000):

        self.capital = start_capital

        for t in tqdm(np.arange(window, self.number_of_bars+1)):

            # get the correct data window
            dates = self.date_range[t-window:t]
            df = self.data.loc[:, dates, :]
            self.current_date = dates[-1]
            current_bars = df.loc[:, self.current_date, :]

            # calculate the bid and ask prices for order management
            close_prices, spreads, points = current_bars['close'], current_bars['spread'], current_bars['point']
            ask_prices = close_prices + spreads * points
            self.current_prices = pd.DataFrame(
                {'bid': close_prices, 'bid': close_prices, 'ask': ask_prices})

            # tick strategy
            self.on_bar(df, current_bars)
            
            # record relevant statistics
            self.capital_record.loc[len(self.capital_record)] = [
                self.current_date, self.capital]

        self.capital_record = self.capital_record.set_index(
            self.capital_record['time'])

    def is_trading_time(self, symbol):
        return not math.isnan(self.current_prices.loc[symbol]['bid'])

    def performance(self):
        self.capital_record['capital'].plot()
        return self.trade_record


if __name__ == "__main__":
    b = Strategy(date_from = datetime(2020, 1, 1), date_to=datetime(2020, 3, 25), timeframe=Timeframes.D1, symbols_to_test=['#AAPL', '#TSLA'])
    b.run(3)