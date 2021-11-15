from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
#%%
# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150
#%%
# In this notebook we will use x_ for x fundamentals,
# and X is the input matrix we want at the end.
x_=pd.read_csv("Annual_Stock_Price_Fundamentals_Filtered.csv",
               index_col=0)
y_=pd.read_csv("Annual_Stock_Price_Performance_Filtered.csv",
               index_col=0)
#%%
def fixNansInX(x):
    '''
    Takes in x DataFrame, edits it so that important keys
    are 0 instead of NaN.
    '''
    keyCheckNullList = ["Short Term Debt",
                        "Long Term Debt",
                        "Interest Expense, Net",
                        "Income Tax (Expense) Benefit, Net",
                        "Cash, Cash Equivalents & Short Term Investments",
                        "Property, Plant & Equipment, Net",
                        "Revenue",
                        "Gross Profit",
                        "Total Current Liabilities",
                        "Property, Plant & Equipment, Net"]
    for i in keyCheckNullList:
        x[i] = x[i].fillna(0)


def addColsToX(x):
    '''
    Takes in x DataFrame, edits it to include:
        Enterprise Value.
        Earnings before interest and tax.

    '''
    x["EV"] = x["Market Cap"] \
              + x["Long Term Debt"] \
              + x["Short Term Debt"] \
              - x["Cash, Cash Equivalents & Short Term Investments"]

    x["EBIT"] = x["Net Income"] \
                - x["Interest Expense, Net"] \
                - x["Income Tax (Expense) Benefit, Net"]
#%%
# Make new X with ratios to learn from.
def getXRatios(x_):
    '''
    Takes in x_, which is the fundamental stock DataFrame raw.
    Outputs X, which is the data encoded into stock ratios.
    '''
    X = pd.DataFrame()

    # EV/EBIT
    X["EV/EBIT"] = x_["EV"] / x_["EBIT"]

    # Op. In./(NWC+FA)
    X["Op. In./(NWC+FA)"] = x_["Operating Income (Loss)"] \
                            / (x_["Total Current Assets"] - x_["Total Current Liabilities"] \
                               + x_["Property, Plant & Equipment, Net"])

    # P/E
    X["P/E"] = x_["Market Cap"] / x_["Net Income"]

    # P/B
    X["P/B"] = x_["Market Cap"] / x_["Total Equity"]

    # P/S
    X["P/S"] = x_["Market Cap"] / x_["Revenue"]

    # Op. In./Interest Expense
    X["Op. In./Interest Expense"] = x_["Operating Income (Loss)"] \
                                    / - x_["Interest Expense, Net"]

    # Working Capital Ratio
    X["Working Capital Ratio"] = x_["Total Current Assets"] \
                                 / x_["Total Current Liabilities"]

    # Return on Equity
    X["RoE"] = x_["Net Income"] / x_["Total Equity"]

    # Return on Capital Employed
    X["ROCE"] = x_["EBIT"] \
                / (x_["Total Assets"] - x_["Total Current Liabilities"])

    # Debt/Equity
    X["Debt/Equity"] = x_["Total Liabilities"] / x_["Total Equity"]

    # Debt Ratio
    X["Debt Ratio"] = x_["Total Assets"] / x_["Total Liabilities"]

    # Cash Ratio
    X["Cash Ratio"] = x_["Cash, Cash Equivalents & Short Term Investments"] \
                      / x_["Total Current Liabilities"]

    # Asset Turnover
    X["Asset Turnover"] = x_["Revenue"] / \
                          x_["Property, Plant & Equipment, Net"]

    # Gross Profit Margin
    X["Gross Profit Margin"] = x_["Gross Profit"] / x_["Revenue"]

    ### Altman ratios ###
    # (CA-CL)/TA
    X["(CA-CL)/TA"] = (x_["Total Current Assets"] \
                       - x_["Total Current Liabilities"]) \
                      / x_["Total Assets"]

    # RE/TA
    X["RE/TA"] = x_["Retained Earnings"] / x_["Total Assets"]

    # EBIT/TA
    X["EBIT/TA"] = x_["EBIT"] / x_["Total Assets"]

    # Book Equity/TL
    X["Book Equity/TL"] = x_["Total Equity"] / x_["Total Liabilities"]

    X.fillna(0, inplace=True)
    return X


def fixXRatios(X):
    '''
    Takes in X, edits it to have the distributions clipped.
    The distribution clippings are done manually by eye,
    with human judgement based on the information.
    '''
    X["RoE"].clip(-5, 5, inplace=True)
    X["Op. In./(NWC+FA)"].clip(-5, 5, inplace=True)
    X["EV/EBIT"].clip(-500, 500, inplace=True)
    X["P/E"].clip(-1000, 1000, inplace=True)
    X["P/B"].clip(-50, 100, inplace=True)
    X["P/S"].clip(0, 500, inplace=True)
    X["Op. In./Interest Expense"].clip(-600, 600, inplace=True)  # -600, 600
    X["Working Capital Ratio"].clip(0, 30, inplace=True)
    X["ROCE"].clip(-2, 2, inplace=True)
    X["Debt/Equity"].clip(0, 100, inplace=True)
    X["Debt Ratio"].clip(0, 50, inplace=True)
    X["Cash Ratio"].clip(0, 30, inplace=True)
    X["Gross Profit Margin"].clip(0, 1, inplace=True)  # how can be >100%?
    X["(CA-CL)/TA"].clip(-1.5, 2, inplace=True)
    X["RE/TA"].clip(-20, 2, inplace=True)
    X["EBIT/TA"].clip(-2, 1, inplace=True)
    X["Book Equity/TL"].clip(-2, 20, inplace=True)
    X["Asset Turnover"].clip(-2000, 2000, inplace=True)  # 0, 500
#%%
def getYPerf(y_):
    '''
    Takes in y_, which has the stock prices and their respective
    dates they were that price.
    Returns a DataFrame y containing the ticker and the
    relative change in price only.
    '''
    y=pd.DataFrame()
    y["Ticker"] = y_["Ticker"]
    y["Perf"]=(y_["Open Price2"]-y_["Open Price"])/y_["Open Price"]
    y["Perf"].fillna(0, inplace=True)
    return y
#%%
from scipy.stats import zscore
def ZscoreSlice(ZscoreSliceVal):
    '''
    Slices the distribution acording to Z score.
    Any values with Z score above/below the argument will be given the max/min Z score value
    '''
    xz=x.apply(zscore) # Dataframe of Z scores
    for key in x.keys():
        xps=ZscoreSliceVal * x[key].std()+x[key].mean()
        xns=ZscoreSliceVal * -x[key].std()+x[key].mean()
        x[key][xz[key]>ZscoreSliceVal]=xps
        x[key][xz[key]<-ZscoreSliceVal]=xns
    return x
#%% RUN THE FUNCTIONS
# From x_ (raw fundamental data) get X (stock fundamental ratios)
fixNansInX(x_)
addColsToX(x_)
X=getXRatios(x_)
fixXRatios(X)

# From y_(stock prices/dates) get y (stock price change)
y=getYPerf(y_)
#%% SEE DISTRIBUTIONS
# See one of the distributions
k=X.keys()[2] # Try different numbers, 0-14.
X[k].hist(bins=100, figsize=(6,5))
plt.title(k);
plt.show()
#%% make a plot of the distributions
# Make a plot of the distributions.
cols, rows = 3, 5
plt.figure(figsize=(5*cols, 5*rows))

for i in range(1, cols*rows):
    if i<len(X.keys()):
        plt.subplot(rows, cols, i)
        k=X.keys()[i]
        X[k].hist(bins=100)
        plt.title(k);
plt.show()
#%% save to csv
y.to_csv("Annual_Stock_Price_Performance_Percentage.csv")
X.to_csv("Annual_Stock_Price_Fundamentals_Ratios.csv")
#%% power transformer to ensure data has good distribution
# Write code to plot out all distributions of X in a nice diagram
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer()
X_t = pd.DataFrame(transformer.fit_transform(X), columns=X.keys())


def plotFunc(n, myDatFrame):
    myKey = myDatFrame.keys()[n]
    plt.hist(myDatFrame[myKey], density=True, bins=30)
    plt.grid()
    plt.xlabel(myKey)
    plt.ylabel('Probability')


plt.figure(figsize=(13, 20))
plotsIwant = [4, 6, 9, 10, 11]

j = 1
for i in plotsIwant:
    plt.subplot(len(plotsIwant), 2, 2 * j - 1)
    plotFunc(i, X)
    if j == 1:
        plt.title('Before Transformation', fontsize=17)
    plt.subplot(len(plotsIwant), 2, 2 * j)
    plotFunc(i, X_t)
    if j == 1:
        plt.title('After Transformation', fontsize=17)
    j += 1

plt.savefig('Transformat_Dists.png', dpi=300)