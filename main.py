import datetime as dt
import numpy as np
import pandas as pd
import simfin as sf
import sys
#%%
def getXDataMerged():
    # this function will download required data and merge into one dataframe

    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    a = sf.load_income(variant='annual', market='us')

    # download balance sheet
    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    b = sf.load_balance(variant='annual', market='us')

    # download cashflow data
    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    c = sf.load_cashflow(variant='annual', market='us')

    # download derived figures and ratios
    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    d = sf.load_derived(variant='annual', market='us')

    # Download key ratios
    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    print('Income Statement DF is: ', a.shape)
    print('Balance Sheet DF is: ', b.shape)
    print('Cash Flow Statement DF is: ', c.shape)
    print('Derived Figures DF is: ', d.shape)

    # merge the pulled dataframes into one dataframe (removing duplicated columns before merging)
    diff_cols = b.columns.difference(a.columns)
    result = pd.merge(a, b[diff_cols], left_index=True, right_index=True)

    diff_cols = c.columns.difference(result.columns)
    result = pd.merge(result, c[diff_cols], left_index=True, right_index=True)

    diff_cols = d.columns.difference(result.columns)
    result = pd.merge(result, d[diff_cols], left_index=True, right_index=True)

    print('merged X data matrix shape is: ', result.shape)

    return result
#%% function for creating y vector (10k to 10k stock performance)
def getYRawData():
    # Retrieve shareprice data into a dataframe
    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    e = sf.load_shareprices(variant='daily', market='us')
    print('Stock Price data matrix is: ', e.shape)
    return e
#%% function to return just the y price and volume near date (as we have imperfect data)
# want the volume data returned to remove stocks with near 0 volume later
# e is the raw y data
# modifier modifies the date span to look between
def getYPriceDataNearDate(ticker, date, modifier, e):
    windowDays = 5
    rows = \
        e[(e["Date"].between(\
            pd.to_datetime(date) + pd.Timedelta(days=modifier),\
            pd.to_datetime(date) + pd.Timedelta(days=windowDays+modifier)))\
            & (e['Ticker']==ticker)]
    if rows.empty:
        return [ticker, np.float('NaN'),\
                np.datetime64('NaT'),\
                np.float('NaN')]
    else: #take the first item of the list of days that fall in the window of accepted days
        return [ticker, rows.iloc[0]['Open'],\
                rows.iloc[0]['Date'],\
                rows.iloc[0]['Volume']*rows.iloc[0]['Open']]
#%%
x = getXDataMerged()
x.reset_index(inplace=True, drop=False)
e = getYRawData()
e.reset_index(inplace=True, drop=False)

#%%
print(getYPriceDataNearDate('AAPL','2012-05-12',30,e))