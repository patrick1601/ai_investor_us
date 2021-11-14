import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
#%%
x_ = pd.read_csv('Annual_Stock_Price_Fundamentals_Filtered.csv', index_col=0)
y_ = pd.read_csv('Annual_Stock_Price_Performance_Filtered.csv', index_col=0)
#%%
keyCheckNullList = ['Short Term Debt',
                    'Long Term Debt',
                    'Interest Expense, Net',
                    'Income Tax (Expense) Benefit, Net',
                    'Cash, Cash Equivalents & Short Term Investments',
                    'Property, Plant & Equipment, Net',
                    'Revenue',
                    'Gross Profit']
#%%
def fixNansInX():
    for key in x_.keys():
        if key in keyCheckNullList:
            x_.loc[x_[key].isnull(), key] = 0
#%% function to create preliminary columns for feature engineering
def addColsToX():
    x_['EV'] = x_['Market Cap'] + x_['Long Term Debt'] + x_['Short Term Debt']\
               - x_['Cash, Cash Equivalents & Short Term Investments']
    x_['EBIT'] = x_['Net Income'] - x_['Interest Expense, Net'] - x_['Income Tax (Expense) Benefit, Net']
#%% function to create new x dataframe with ratios
def getXRatios():
    x = pd.DataFrame()

    # EV/EBIT
    x['EV/EBIT'] = x_['EV'] / x_['EBIT']

    # EV/EBITDA
    x['EV/EBITDA'] = x_['EV'] / x_['EBITDA']

    # Op. In./(NWC+FA)
    x['Op. In./(NWC+FA)'] = x_['Operating Income (Loss)'] / (x_['Total Current Assets'] -
                                                             x_['Total Current Liabilities'] +
                                                             x_['Property, Plant & Equipment, Net'])

    # P/E
    x['P/E'] = x_['Market Cap'] / x_['Net Income']

    # P/B
    x['P/B'] = x_['Market Cap'] / x_['Total Equity']

    # P/S
    x['P/S'] = x_['Market Cap'] / x_['Revenue']

    # Op. In./Interest Expense
    x['Op. In./Interest Expense'] = x_['Operating Income (Loss)'] / - x_['Interest Expense, Net']

    # Working Capital Ratio
    x['Working Capital Ratio'] = x_['Total Current Assets'] / x_['Total Current Liabilities']

    # Return on Equity
    x['RoE'] = x_['Net Income'] / x_['Total Equity']

    # Return on Capital Employed
    x['ROCE'] = x_['EBIT'] / (x_['Total Assets'] - x_['Total Current Liabilities'])

    # Debt/Equity
    x['Debt/Equity'] = x_['Total Liabilities'] / x_['Total Equity']

    # Debt Ratio
    x['Debt Ratio'] = x_['Total Assets'] / x_['Total Liabilities']

    # Cash Ratio
    x['Cash Ratio'] = x_['Cash, Cash Equivalents & Short Term Investments'] / x_['Total Current Liabilities']

    # Asset Turnover
    x['Asset Turnover'] = x_['Revenue'] / x_['Property, Plant & Equipment, Net']

    # Gross Profit Margin
    x['Gross Profit Margin'] = x_['Gross Profit'] / x_['Revenue']

    # Current ratio
    x['Current Ratio'] = x_['Current Ratio']

    # Dividend yield
    x['Dividend Yield'] = (x_['Dividends Per Share'] * x_['Shares (Diluted)']) / x_['Market Cap']

    # Pietroski F-Score
    x['Pietroski F-Score'] = x_['Pietroski F-Score']

    ### Altman Ratios ###
    # (CA-CL)/TA
    x['(CA-CL)/TA'] = (x_['Total Current Assets'] - x_['Total Current Liabilities'])/x_['Total Assets']

    # RE/TA
    x['RE/TA'] = x_['Retained Earnings']/x_['Total Assets']

    # EBIT/TA
    x['EBIT/TA'] = x_['EBIT']/x_['Total Assets']

    # Book Equity/TL
    x['Book Equity/TL'] = x_['Total Equity']/x_['Total Liabilities']

    return x

#%% function to scale data
def maxMinRatio(m, text, max, min):
    m.loc[x[text]>max,text]=max
    m.loc[x[text]<min,text]=min
#%% manual customized ratio cutoffs
def fixXRatios():
    for key in x.keys():
        x[key][x[key].isnull()]=0
        if ((key == 'RoE') or (key == 'Op. In./(NWC+FA)')):
            maxMinRatio(x, key, 5, -5)
        elif (key == 'EV/EBIT'):
            maxMinRatio(x, key, 500, -500)
        elif (key == 'EV/EBITDA'):
            maxMinRatio(x, key, 500, -500)
        elif (key == 'P/E'):
            maxMinRatio(x, key, 1000, -1000)
        elif (key == 'P/B'):
            maxMinRatio(x, key, 100, -50)
        elif (key == 'P/S'):
            maxMinRatio(x, key, 500, 0)
        elif (key == 'Op. In./Interest Expense'):
            maxMinRatio(x, key, 800, -200)
        elif (key == 'Working Capital Ratio'):
            maxMinRatio(x, key, 30, 0)
        elif (key == 'ROCE'):
            maxMinRatio(x, key, 2, 0)
        elif (key == 'Debt/Equity'):
            maxMinRatio(x, key, 100, 0)
        elif (key == 'Debt Ratio'):
            maxMinRatio(x, key, 50, 0)
        elif (key == 'Cash Ratio'):
            maxMinRatio(x, key, 30, 0)
        elif (key == 'Gross Profit Margin'):
            maxMinRatio(x, key, 3, 0)
        elif (key == 'Current Ratio'):
            maxMinRatio(x, key, 10, 0)
        elif (key == 'Div Y'):
            maxMinRatio(x, key, 1, 0)
        # Altman Ratios
        elif (key == '(CA-CL)/TA'):
            maxMinRatio(x, key, 2, -1.5)
        elif (key == 'RA/TA'):
            maxMinRatio(x, key, 2, -20)
        elif (key == 'EBIT/TA'):
            maxMinRatio(x, key, 1, -2)
        elif (key == 'Book Equity/TL'):
            maxMinRatio(x, key, 20, -2)
        else:
            maxMinRatio(x, key, 2000, -2000)
    return x
#%% run main program to create training dataset
fixNansInX()
addColsToX()
x = getXRatios()
x = fixXRatios()
#%% save to csv
x.to_csv('Annual_Stock_Price_Fundamentals_Ratios.csv')
#%% function to create final y vector by finding percent change between the start and end prices for each period
def getYPerf(y_):
    y = pd.DataFrame()
    y['Ticker'] = y_['Ticker']
    y['Perf'] = (y_['Open Price2'] - y_['Open Price'])/y_['Open Price']
    y[y['Perf'].isnull()] = 0
    return y
#%% run main program to create y vector
y = getYPerf(y_)
#%% save to csv
y.to_csv('Annual_Stock_Price_Performance_Percentage.csv')
#%% code to plot out all distributions of X
transformer = PowerTransformer()
x_t = pd.DataFrame(transformer.fit_transform(x), columns=x.keys())

def plotFunc(n, myDatFrame):
    myKey = myDatFrame.keys()[n]
    plt.hist(myDatFrame[myKey], density=True, bins=30)
    plt.grid()
    plt.xlabel(myKey)
    plt.ylabel('Probability')

plt.figure(figsize=(13,20))

plotsIwant=[4, 6, 9, 10, 11]
j = 1
for i in plotsIwant:
    plt.subplot(len(plotsIwant), 2, 2*j-1)
    plotFunc(i, x)
    if j == 1:
        plt.title('Before Transformation', fontsize=17)
    plt.subplot(len(plotsIwant), 2, 2*j)
    plotFunc(i,x_t)
    if j == 1:
        plt.title('After Transformation', fontsize=17)
    j += 1
plt.savefig('Transformat_Dists.png', dpi=300)