import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#%% load x and y dataframes from csv files
x = pd.read_csv('Annual_Stock_Price_Fundamentals.csv')
y = pd.read_csv('Annual_Stock_Price_Performance.csv')
#%% create scatter matrix
attributes = ['Revenue', 'Net Income']
scatter_matrix(x[attributes])
plt.show()
#%% show tickers with volumes under 10000