import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#%% load x and y dataframes from csv files
x = pd.read_csv('Annual_Stock_Price_Fundamentals.csv', index_col=0)
y = pd.read_csv('Annual_Stock_Price_Performance.csv', index_col=0)
#%% create scatter matrix
attributes = ['Revenue', 'Net Income']
scatter_matrix(x[attributes])
plt.show()
#%% show tickers with volumes under 10000
low_volume_stocks = y[(y['Volume'] < 1e4) | (y['Volume2'] < 1e4)]
#%% pre process data
# remove rows without share prices
bool_list = ~y['Open Price'].isnull()
y = y[bool_list]
x = x[bool_list]

# remove rows where number of shares is null
bool_list = ~x['Shares (Diluted)'].isnull()
y = y[bool_list]
x = x[bool_list]

# remove rows where there is low or no volume
bool_list = ~((y['Volume']<1e4) | (y['Volume2']<1e4))
y = y[bool_list]
x = x[bool_list]

# remove rows where dates are missing also removes latest data which we can't use
bool_list = ~y["Date2"].isnull()
y = y[bool_list]
x = x[bool_list]

y=y.reset_index(drop=True)
x=x.reset_index(drop=True)

#Create a market cap column
x["Market Cap"] = y["Open Price"]*x["Shares (Diluted)"]

print(x.shape)
print(y.shape)

# Save to CSV
x.to_csv("Annual_Stock_Price_Fundamentals_Filtered.csv")
y.to_csv("Annual_Stock_Price_Performance_Filtered.csv")