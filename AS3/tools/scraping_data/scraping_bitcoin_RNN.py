#!/usr/bin/env python
import datetime
import sys
import os
import pandas as pd

file_location = os.path.abspath(__file__)  # Get current file abspath
root_directory = os.path.dirname(file_location)  # Get root dir
sys.path.append(os.path.join(root_directory))
from binance.spot import Spot as Client
from utils.secrets import *

# Set up client connector
# RSA keys set up
[api_key, api_secret] = get_api_key()
client = Client(api_key, api_secret)
# Set up params
INTERVAL = "1h"  # Interval Binance API query
INTERVAL_MS = int(60 * 60 * 1e6)
BINANCE_HEADER = ["Open time", "Open price", "High price", "Low price", "Close price", "Volume", "Close time",
                  "Quote asset volume", "Number of trades", "Taker buy base asset vol", "Taker buy quote asset vol",
                  "Ignore"]  # Dataframe header


class Date:
    """
    Store a date, transform into ms timestamp
    """

    def __init__(self, year, month, day, hour, min=None):
        self.y = year
        self.m = month
        self.d = day
        self.h = hour
        self.min = 0 if min is None else min

    def get_timestamp(self):
        return int(float(datetime.datetime(self.y, self.m, self.d, self.h, self.min).strftime('%s.%f')) * 1000)

    def print(self):
        print("{}/{:02d}/{:02d} - {:02d}:{:02d}".format(self.y, self.m, self.d, self.h, self.min))


def get_klines(start, end, pair) -> pd.DataFrame:
    """
    Fetch Binance Klines. Start and end date must represent at most one thousand interval time.

    Arguments :
        start -- Date
        end   -- Date

    Return
        df    -- pd.DataFrame
    """
    return pd.DataFrame(client.klines(pair, INTERVAL, limit=1000, startTime=start, endTime=end), columns=BINANCE_HEADER)


def _check_data(df) -> pd.DataFrame:
    """
    Check and fix Open time data .

    Arguments :
        df    -- pd.DataFrame

    Return
        df    -- pd.DataFrame

    """

    for i in range(len(df.index) - 1):
        o1 = df["Open time"][i] + int(INTERVAL_MS / 1e3)
        o2 = df["Open time"][i + 1]

        if o1 != o2:  # Check if the i and i+1 data are consistent
            s = df.copy().xs(i)
            for inter in range(o1, o2, int(INTERVAL_MS / 1e3)):  # Loop to prevent multiple missing data
                s["Open time"] = inter
                for c in ["Volume", "Quote asset volume", "Number of trades", "Taker buy base asset vol",
                          "Taker buy quote asset vol", "Close time"]:
                    if c in df.columns and c != "Close time":
                        s[c] = 0
                    elif c in df.columns and c == "Close time":
                        s[c] = inter - 1 + int(INTERVAL_MS / 1e3)
                df = pd.concat([df[:i], pd.DataFrame(dict(s), columns=df.columns, index=[len(df.columns)]), df[i:]])

    return df.sort_values(by=["Open time"]).reset_index(drop=True)


def get_data_klines(start, end, pair) -> pd.DataFrame:
    """
    Multiple get_klines calls (caused by the API limitation) to fetch Binance Klines data from start date to end date.

    Arguments :
        start -- Date
        end   -- Date

    Return
        df    -- pd.DataFrame
    """
    s = start.get_timestamp()
    e = end.get_timestamp()
    df = pd.DataFrame([], columns=BINANCE_HEADER)

    for t in range(s, e, INTERVAL_MS):
        if t + INTERVAL_MS <= e:
            df = pd.concat([df, get_klines(t, t + INTERVAL_MS, pair)], ignore_index=True)
        else:
            df = pd.concat([df, get_klines(t, e, pair)], ignore_index=True)

    return _check_data(df.drop_duplicates(subset=["Open time"]).reset_index(drop=True))


def test_time_dfs(df1, df2):
    """
    Checking the correlation between each "Open time" data from two dfs.
    Print the row where the error happen if there is one.

    Arguments :
        df1   -- pd.DataFrame
        df2   -- pd.DataFrame

    """
    size = len(df2.index) if len(df2.index) < len(df1.index) else len(df1.index)
    for i in range(size):
        o1 = int(df1["Open time"][i])
        o2 = int(df2["Open time"][i])
        if o1 != o2:
            print(i, o1, o2)
            break


def formatting(df, cols) -> pd.DataFrame:
    """
    Formatting the final DataFrame. Dfs is a list with one or multiple DataFrame to be concatenated.
    Cols is a list of the columns required in the final DataFrame.

    Arguments :
        dfs     -- pd.DataFrame
        cols    -- list
    Return
        df      -- pd.DataFrame

    """
    final_df = pd.DataFrame()

    for col in df.columns:
        if col in cols and col not in final_df.columns:
            final_df[col] = df[col].copy()

    return final_df


def create_dataset(start: Date, end: Date, pair=None, path=None) -> pd.DataFrame:
    if pair is None:
        pair = PAIR # Default is BTCUSDT if no params
    df_klines = get_data_klines(start, end, pair)

    df = formatting(df_klines, ["Open time", "Open price", "High price", "Low price", "Close price", "Volume",
                                "Number of trades"])
    df.to_excel(path, index=False)
    print(f'Exported to {path}')

    return df
