import datetime as dt
import simfin as sf
import pandas as pd
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
#%%
X = getXDataMerged()