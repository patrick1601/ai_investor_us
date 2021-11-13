import datetime as dt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
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

    result.reset_index(inplace=True, drop=False)

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
    e.reset_index(inplace=True, drop=False)
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
        return [ticker, np.float64('NaN'),\
                np.datetime64('NaT'),\
                np.float64('NaN')]
    else: #take the first item of the list of days that fall in the window of accepted days
        return [ticker, rows.iloc[0]['Open'],\
                rows.iloc[0]['Date'],\
                rows.iloc[0]['Volume']*rows.iloc[0]['Open']]
#%% function that shows the close prices for the report dates in the x dataframe
# modifier is the hold period for a stock
# x is the vector of company data
# e is the Y raw data (stock prices and date for all days)
def getYPricesReportDateAndTargetDate(x, e, modifier=365):
    i = 0

    # Preallocation list of list of 2
    # [(price at date) (price at date + modifier)]
    y = [[None]*8 for i in range (len(x))]

    # the performance date from->to.
    # Want this to be publish date because of the time lag between report date (which can't be actioned on)
    # and publish date (public)
    whichDateCol = 'Publish Date'

    for index in range(len(x)):
        y[i] = (getYPriceDataNearDate(\
            x['Ticker'].iloc[index], x[whichDateCol].iloc[index], 0, e) + \
                getYPriceDataNearDate(\
                    x['Ticker'].iloc[index], x[whichDateCol].iloc[index], modifier, e))
        i=i+1
    return y

#%%
x = getXDataMerged()
x.to_csv("Annual_Stock_Price_Fundamentals.csv")
#%% takes several hours
e = getYRawData()
y = getYPricesReportDateAndTargetDate(x, e, 365)
#%% save y to csv file
y=pd.DataFrame(y, columns=['Ticker', 'Open Price', 'Date', 'Volume', 'Ticker2', 'Open Price2', 'Date2', 'Volume2'])
y.to_csv('Annual_Stock_Price_Performance.csv')