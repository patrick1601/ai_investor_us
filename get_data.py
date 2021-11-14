import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import simfin as sf
#%%
# Set the plotting DPI settings to be higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150
#%%
def getXDataMerged():
    # this function will download required data and merge into one dataframe

    # Set your SimFin+ API-key for downloading data.
    sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')
    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')
    # download income statement
    incomeStatementData = sf.load_income(variant='annual', market='us')
    # download balance sheet
    balanceSheetData = sf.load_balance(variant='annual', market='us')
    # download cashflow data
    cashFlowData = sf.load_cashflow(variant='annual', market='us')
    # download derived ratios
    derivedData = sf.load_derived(variant='annual', market='us')

    print('Income Statement DF is: ', incomeStatementData.shape)
    print('Balance Sheet DF is: ', balanceSheetData.shape)
    print('Cash Flow Statement DF is: ', cashFlowData.shape)
    print('Derived Figures DF is: ', derivedData.shape)

    # merge the pulled dataframes into one dataframe (removing duplicated columns before merging)
    diff_cols = balanceSheetData.columns.difference(incomeStatementData.columns)
    result = pd.merge(incomeStatementData, balanceSheetData[diff_cols], left_index=True, right_index=True)

    diff_cols = cashFlowData.columns.difference(result.columns)
    result = pd.merge(result, cashFlowData[diff_cols], left_index=True, right_index=True)

    diff_cols = derivedData.columns.difference(result.columns)
    result = pd.merge(result, derivedData[diff_cols], left_index=True, right_index=True)

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
    dailySharePrices = sf.load_shareprices(variant='daily', market='us')
    dailySharePrices.reset_index(inplace=True, drop=False)
    print('Stock Price data matrix is: ', dailySharePrices.shape)
    return dailySharePrices
#%% function to return just the y price and volume near date (as we have imperfect data)
# want the volume data returned to remove stocks with near 0 volume later
# e is the raw y data
# modifier modifies the date span to look between
def getYPriceDataNearDate(ticker, date, modifier, dailySharePrices):
    '''
    Return just the y price and volume.
    Take the first day price/volume of the list of days,
    that fall in the window of accepted days.
    'modifier' just modifies the date to look between.
    Returns a list.
    '''
    windowDays = 5
    rows = dailySharePrices[
        (dailySharePrices["Date"].between(pd.to_datetime(date)
                                          + pd.Timedelta(days=modifier),
                                          pd.to_datetime(date)
                                          + pd.Timedelta(days=windowDays
                                                              + modifier)
                                          )
         ) & (dailySharePrices["Ticker"] == ticker)]

    if rows.empty:
        return [ticker, np.float("NaN"),
                np.datetime64('NaT'),
                np.float("NaN")]
    else:
        return [ticker, rows.iloc[0]["Open"],
                rows.iloc[0]["Date"],
                rows.iloc[0]["Volume"] * rows.iloc[0]["Open"]]
#%%
d=getYRawData()
#%%
print(getYPriceDataNearDate('AAPL', '2012-05-12', 0, d))
#%%
print(getYPriceDataNearDate('AAPL', '2012-05-12', 30, d))
#%% function that shows the close prices for the report dates in the x dataframe
# modifier is the hold period for a stock
# x is the vector of company data
# e is the Y raw data (stock prices and date for all days)
def getYPricesReportDateAndTargetDate(x, e, modifier=365):
    '''
    Takes in all fundamental data X, all stock prices over time y,
    and modifier (days), and returns the stock price info for the
    data report date, as well as the stock price one year from that date
    (if modifier is left as modifier=365)
    '''
    # Preallocation list of list of 2
    # [(price at date) (price at date + modifier)]
    y = [[None] * 8 for i in range(len(x))]

    whichDateCol = 'Publish Date'  # or 'Report Date',
    # is the performance date from->to. Want this to be publish date.

    # Because of time lag between report date
    # (which can't be actioned on) and publish date
    # (data we can trade with)

    # In the end decided this instead of iterating through index.
    # Iterate through a range rather than index, as X might not have
    # monotonic increasing index 1, 2, 3, etc.
    i = 0
    for index in range(len(x)):
        y[i] = (getYPriceDataNearDate(x['Ticker'].iloc[index],
                                      x[whichDateCol].iloc[index], 0, d)
                + getYPriceDataNearDate(x['Ticker'].iloc[index],
                                        x[whichDateCol].iloc[index],
                                        modifier, d))
        i = i + 1

    return y
#%%
X = getXDataMerged()
X.to_csv("Annual_Stock_Price_Fundamentals.csv")
#%%
print(X.keys())
#%%
d = getYRawData()
print(d[d['Ticker']=='GOOG'])
#%%
# We want to know the performance for each stock, each year, between 10-K report dates.
# takes VERY long time, several hours,
y = getYPricesReportDateAndTargetDate(X, d, 365) # because of lookups in this function.
#%% save y to csv file
y = pd.DataFrame(y, columns=['Ticker', 'Open Price', 'Date', 'Volume',
                             'Ticker2', 'Open Price2', 'Date2', 'Volume2'
                            ])
y.to_csv("Annual_Stock_Price_Performance.csv")