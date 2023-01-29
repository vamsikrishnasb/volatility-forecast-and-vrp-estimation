import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import glob
import warnings
from datetime import datetime
from nsepython import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.simplefilter("ignore")
next_trading_date = '2023-01-19'

folder_path = 'your/folder/path/here/'
path = folder_path + 'sample_input_vol.csv'
df = pd.read_csv(path)

vol_df = df[['date', 'close_ul', 'log_returns_ul_c2c', 'vol_c2c_20d']]
_lambda = 0.94
n = len(vol_df[vol_df['date'] <= '2015-12-30'])
vol_df['log_returns_squared'] = vol_df['log_returns_ul_c2c'] ** 2
vol_df['ewma_variance'] = ''
vol_df['ewma_daily_vol'] = ''
vol_df['ewma_annual_vol'] = ''
vol_df['initial_weights'] = ''
vol_df['weights_times_log_returns_squared'] = ''

# Calculate EWMA volatility for 2016-01-01
for i in range(0, n + 1):
    vol_df['initial_weights'][i] = (1- _lambda) * _lambda ** (n - i)
vol_df['log_returns_ul_c2c'] = pd.to_numeric(vol_df['log_returns_ul_c2c'], errors='coerce')
vol_df['initial_weights'] = pd.to_numeric(vol_df['initial_weights'], errors='coerce')
vol_df['weights_times_log_returns_squared'] = vol_df['initial_weights'] * vol_df['log_returns_squared']
initial_ewma_var = vol_df['weights_times_log_returns_squared'].sum()
initial_ewma_daily_vol = initial_ewma_var ** 0.5
initial_ewma_annual_vol = (256 ** 0.5) * (initial_ewma_var ** 0.5)
vol_df.loc[vol_df['date'] == '2015-12-31', 'ewma_variance'] = initial_ewma_var
vol_df.loc[vol_df['date'] == '2015-12-31', 'ewma_daily_vol'] = initial_ewma_daily_vol
vol_df.loc[vol_df['date'] == '2015-12-31', 'ewma_annual_vol'] = initial_ewma_annual_vol

# Use the EWMA volatility calculated for 2016-01-01 to calculate the 
# EWMA volatility for the subsequent dates
m = len(vol_df[vol_df['date'] <= '2015-12-31'])
vol_df['ewma_variance'] = pd.to_numeric(vol_df['ewma_variance'], errors='coerce')
vol_df['log_returns_squared'] = pd.to_numeric(vol_df['log_returns_squared'], errors='coerce')
for j in range(m, len(vol_df)):
    vol_df['ewma_variance'][j] = _lambda * vol_df['ewma_variance'][j - 1] + (1 - _lambda) * vol_df['log_returns_squared'][j]
vol_df['ewma_daily_vol'] = vol_df['ewma_variance'] ** 0.5
vol_df['ewma_annual_vol'] = (256 ** 0.5) * vol_df['ewma_daily_vol']
daily_vol_forecast = vol_df['ewma_daily_vol'].iloc[-1]
annual_vol_forecast = vol_df['ewma_annual_vol'].iloc[-1]
print("Daily Vol Forecast for", next_trading_date, ":", round(daily_vol_forecast * 100, 2), "%")
print("Annual Vol Forecast for", next_trading_date, ":", round(annual_vol_forecast * 100, 2), "%")

# Fetch options data to get ATM IV
path = folder_path + 'sample_input_options.csv'
df = pd.read_csv(path)

options_df = df[
    (df['days_to_expiry'].between(3, 14))
    & (df['put_close'] != 0)
    & (df['call_close'] != 0)
    & (df['implied_volatility'] >= 0.01)
]

options_df['settle_price_option'] = pd.to_numeric(options_df['settle_price_option'], errors='coerce')
options_df['close_ul'] = pd.to_numeric(options_df['close_ul'], errors='coerce')
options_df['futures_price'] = options_df['forward_close']
options_df['r'] = options_df['rf_rate']
options_df['straddle_price'] = options_df['put_close'] + options_df['call_close']
options_df['s_minus_k_abs'] = (options_df['futures_price'] - options_df['strike']).abs()

temp = options_df.groupby(['date'], sort=False)['s_minus_k_abs'].min()
temp = pd.DataFrame(temp)
options_df = options_df[[
    'date', 'days_to_expiry', 's_minus_k_abs', 'strike', 'implied_volatility', 
    'straddle_price', 'close_ul', 'futures_price', 'rf_rate'
]]
options_df = temp.merge(options_df, on=['date', 's_minus_k_abs'], how='inner')
options_df['straddle_price'] = pd.to_numeric(options_df['straddle_price'], errors='coerce')

temp = options_df.groupby(['date'], sort=False)['days_to_expiry'].min()
temp = pd.DataFrame(temp)
options_df = temp.merge(options_df, on=['date', 'days_to_expiry'], how='inner')

options_df = options_df[[
    'date', 'close_ul', 'futures_price', 'rf_rate', 'days_to_expiry', 'straddle_price',
    'implied_volatility'
]]

options_df['atm_iv'] = options_df['implied_volatility']

ewma_vol_atm_iv_df = options_df.merge(vol_df, on=['date'], how='inner')
ewma_vol_atm_iv_df = ewma_vol_atm_iv_df[['date', 'ewma_annual_vol', 'vol_c2c_20d', 'atm_iv']]

#hav: historical actual volatility
ewma_vol_atm_iv_df['date_hav'] = ewma_vol_atm_iv_df['date']
ewma_vol_atm_iv_df['min_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].min()
ewma_vol_atm_iv_df['_10pc_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].quantile(0.1)
ewma_vol_atm_iv_df['_25pc_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].quantile(0.25)
ewma_vol_atm_iv_df['_50pc_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].quantile(0.5)
ewma_vol_atm_iv_df['_75pc_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].quantile(0.75)
ewma_vol_atm_iv_df['_90pc_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].quantile(0.9)
ewma_vol_atm_iv_df['max_hav'] = ewma_vol_atm_iv_df['vol_c2c_20d'].max()

#hav: forecasted actual volatility
ewma_vol_atm_iv_df['date_fav'] = ewma_vol_atm_iv_df['date']
ewma_vol_atm_iv_df['min_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].min()
ewma_vol_atm_iv_df['_10pc_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].quantile(0.1)
ewma_vol_atm_iv_df['_25pc_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].quantile(0.25)
ewma_vol_atm_iv_df['_50pc_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].quantile(0.5)
ewma_vol_atm_iv_df['_75pc_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].quantile(0.75)
ewma_vol_atm_iv_df['_90pc_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].quantile(0.9)
ewma_vol_atm_iv_df['max_fav'] = ewma_vol_atm_iv_df['ewma_annual_vol'].max()

#iv: implied volatility
ewma_vol_atm_iv_df['date_iv'] = ewma_vol_atm_iv_df['date']
ewma_vol_atm_iv_df['min_iv'] = ewma_vol_atm_iv_df['atm_iv'].min()
ewma_vol_atm_iv_df['_10pc_iv'] = ewma_vol_atm_iv_df['atm_iv'].quantile(0.1)
ewma_vol_atm_iv_df['_25pc_iv'] = ewma_vol_atm_iv_df['atm_iv'].quantile(0.25)
ewma_vol_atm_iv_df['_50pc_iv'] = ewma_vol_atm_iv_df['atm_iv'].quantile(0.5)
ewma_vol_atm_iv_df['_75pc_iv'] = ewma_vol_atm_iv_df['atm_iv'].quantile(0.75)
ewma_vol_atm_iv_df['_90pc_iv'] = ewma_vol_atm_iv_df['atm_iv'].quantile(0.9)
ewma_vol_atm_iv_df['max_iv'] = ewma_vol_atm_iv_df['atm_iv'].max()

plot_df = pd.melt(ewma_vol_atm_iv_df[ewma_vol_atm_iv_df['date'] >= '2016-01-01'], id_vars=['date'], value_vars=['atm_iv', 'vol_c2c_20d'])
fig = px.line(plot_df, x='date', y='value', color='variable', log_y=True, title="ATM IV vs 20D Vol | " + underlying)
fig.show()

plot_df = pd.melt(ewma_vol_atm_iv_df[ewma_vol_atm_iv_df['date'] >= '2016-01-01'], id_vars=['date'], value_vars=['atm_iv', 'ewma_annual_vol'])
fig = px.line(plot_df, x='date', y='value', color='variable', log_y=True, title="ATM IV vs EWMA Vol | " + underlying)
fig.show()

plot_df = pd.melt(ewma_vol_atm_iv_df[ewma_vol_atm_iv_df['date'] >= '2016-01-01'], id_vars=['date'], value_vars=['atm_iv', '_10pc_iv', '_25pc_iv', '_50pc_iv', '_75pc_iv', '_90pc_iv'])
fig = px.line(plot_df, x='date', y='value', color='variable', log_y=True, title="ATM IV | " + underlying)
fig.show()

plot_df = pd.melt(ewma_vol_atm_iv_df[ewma_vol_atm_iv_df['date'] >= '2016-01-01'], id_vars=['date'], value_vars=['vol_c2c_20d', '_10pc_hav', '_25pc_hav', '_50pc_hav', '_75pc_hav', '_90pc_hav'])
fig = px.line(plot_df, x='date', y='value', color='variable', log_y=True, title="20D Vol | " + underlying)
fig.show()

plot_df = pd.melt(ewma_vol_atm_iv_df[ewma_vol_atm_iv_df['date'] >= '2016-01-01'], id_vars=['date'], value_vars=['ewma_annual_vol', '_10pc_fav', '_25pc_fav', '_50pc_fav', '_75pc_fav', '_90pc_fav'])
fig = px.line(plot_df, x='date', y='value', color='variable', log_y=True, title="EWMA Vol | " + underlying)
fig.show()

_20d_c2c_iv_diff = pd.DataFrame(columns = ['date', 'vrp', 'min', '_10pc', '_25pc', '_50pc', '_75pc', '_90pc', 'max'])

_20d_c2c_iv_diff['date'] = ewma_vol_atm_iv_df['date']
_20d_c2c_iv_diff['vrp'] = 1.00 * (ewma_vol_atm_iv_df['atm_iv'] - ewma_vol_atm_iv_df['ewma_annual_vol']) / ewma_vol_atm_iv_df['ewma_annual_vol']
_20d_c2c_iv_diff['min'] = _20d_c2c_iv_diff['vrp'].min()
_20d_c2c_iv_diff['_10pc'] = _20d_c2c_iv_diff['vrp'].quantile(0.1)
_20d_c2c_iv_diff['_25pc'] = _20d_c2c_iv_diff['vrp'].quantile(0.25)
_20d_c2c_iv_diff['_50pc'] = _20d_c2c_iv_diff['vrp'].quantile(0.5)
_20d_c2c_iv_diff['_75pc'] = _20d_c2c_iv_diff['vrp'].quantile(0.75)
_20d_c2c_iv_diff['_90pc'] = _20d_c2c_iv_diff['vrp'].quantile(0.9)
_20d_c2c_iv_diff['max'] = _20d_c2c_iv_diff['vrp'].max()

plot_df = pd.melt(_20d_c2c_iv_diff[_20d_c2c_iv_diff['date'] >= '2016-01-01'], id_vars=['date'], value_vars=['vrp', '_10pc', '_25pc', '_50pc', '_75pc', '_90pc'])
fig = px.line(plot_df, x='date', y='value', color='variable', log_y=True, title="VRP | " + underlying)
fig.show()