from sklearn.feature_selection import VarianceThreshold
import os
import matplotlib.pyplot as plt
import numpy as np
from binance.client import Client
from scipy.stats import pearsonr
from ta import *
import pandas_ta as ta
import seaborn as sns
import pandas as pd
import config


# Get the current absolute path
current_path = os.path.abspath(os.getcwd())
print(current_path)
relative_path = os.path.relpath(current_path)

print(relative_path)


client = Client(config.apiKey, config.apiSecret)
symbols = ['DOGEUSDT', 'BTCUSDT', 'ETHUSDT', 'MATICUSDT']


def add_technical_indicators(df, symbol):
    df = add_all_ta_features(df, open=f'Open_{symbol.replace("USDT", "")}', high=f'High_{symbol.replace("USDT", "")}',
                             low=f'Low_{symbol.replace("USDT", "")}', close=f'Close_{symbol.replace("USDT", "")}', volume=f'Volume_{symbol.replace("USDT", "")}', fillna=True)
    return df


def drop_written_correlated_columns(df, threshold):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(
        upper_tri[column] > threshold)]
    try:
        # to_drop.remove('Close_DOGE')
        to_drop.append('others_dr')
    except ValueError:
        pass
    return to_drop


def candlestick_stats(open_series, high_series, low_series, close_series):
    mean_series = (open_series + high_series + low_series + close_series) / 4
    std_series = np.sqrt(((open_series - mean_series)**2 + (high_series - mean_series)
                         ** 2 + (low_series - mean_series)**2 + (close_series - mean_series)**2) / 4)
    return std_series


def calculate_correlation(series_a, series_b, timeunits):
    volatility = []
    for i in range(timeunits):
        volatility.append(float('nan'))
    for i in range(timeunits, len(series_a)):
        r, p = pearsonr(list(series_a.iloc[i-timeunits:i]),
                        list((series_b.iloc[i-timeunits:i])))
        if float(p) < 0.05 or float(r) == 1:
            r = 0

        volatility.append(round(r, 6))
    return volatility


df = pd.DataFrame()

# loop through symbols and gather data
for symbol in symbols:
    # symbol_df = client.futures_klines(
    #     symbol=symbol, interval=Client.KLINE_INTERVAL_5MINUTE, period='365 days ago')
    symbol_df = client.get_historical_klines(
        symbol, '5m', '365 days ago')
    symbol_df = pd.DataFrame(symbol_df, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    symbol_df.drop(columns=[
        'Close time', 'Quote asset volume', 'Trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True, axis=1)
    symbol_df['Time'] = pd.to_datetime(symbol_df['Time'], unit='ms')
    symbol_df.set_index('Time', inplace=True)
    symbol_df.rename(columns={'Open': f'Open_{symbol.replace("USDT", "")}',
                              'High': f'High_{symbol.replace("USDT", "")}',
                              'Low': f'Low_{symbol.replace("USDT", "")}',
                              'Close': f'Close_{symbol.replace("USDT", "")}',
                              'Volume': f'Volume_{symbol.replace("USDT", "")}'}, inplace=True)
    symbol_df = symbol_df.astype(float)
    df = df.join(symbol_df, how='outer')

    if symbol == 'DOGEUSDT':
        df = add_technical_indicators(df, symbol)


df = df.fillna(df.mean())
# loop through symbols and calculate correlations
for sym in symbols:
    string = str(str(sym).replace('USDT', ''))
    df[f'candlestick_stats_{string}'] = candlestick_stats(
        df[f'Open_{string}'], df[f'High_{string}'], df[f'Low_{string}'], df[f'Close_{string}'])
    if sym == 'DOGEUSDT':
        continue
    df[f'cov_DOGE_{string}'] = calculate_correlation(df['Close_DOGE'],
                                                     df[f'Close_{string}'], 5)
to_drop = drop_written_correlated_columns(df, 0.7)
to_drop.append('Open_DOGE')
to_drop.remove('Close_DOGE')
to_drop.remove('candlestick_stats_DOGE')

var_thres = VarianceThreshold(threshold=0.05)
var_thres.fit(df)

constant_columns = [column for column in df.columns
                    if column not in df.columns[var_thres.get_support()]]

print(len(constant_columns))

for feature in constant_columns:
    print(feature)


constant_columns.remove('Close_DOGE')

df.drop(constant_columns, axis=1)

# Check if the specified columns exist in the DataFrame
columns_to_drop = ['trend_adx', 'trend_aroon_up', 'trend_aroon_down', 'trend_psar_up_indicator',
                   'trend_psar_down_indicator', 'volatility_bbhi', 'volatility_bbli', 'volatility_kchi', 'volatility_kcli']
if set(columns_to_drop).issubset(set(df.columns)):
    df.drop(columns_to_drop, axis=1, inplace=True)


print(df.index)

df = df.drop(to_drop, axis=1)
df = df.fillna(df.mean())


# df.to_excel(f'/root/deep_learning/data/datasets/Features.xlsx', index=True)

datatoexcel = pd.ExcelWriter('/root/deep_learning/data/datasets/Features.xlsx')

# write DataFrame to excel
df.to_excel(datatoexcel)
datatoexcel.save()


# Calculate correlations
corr = df.corr()
print(corr)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Save the plot
plt.savefig(f"/root/deep_learning/data/graphs/correlation_matrix.png")
