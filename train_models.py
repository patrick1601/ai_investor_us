import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
#%% read in data
x = pd.read_csv('Annual_Stock_Price_Fundamentals_Ratios.csv', index_col=0)
y = pd.read_csv('Annual_Stock_Price_Performance_Percentage.csv', index_col=0)
y = y['Perf']
#%%
print('Mean Baseline Return of Stocks: ', y.mean())
print(' ')
#%% train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)
print(' ')
#%% linear regression model
pl_linear = Pipeline([('Power Transformer', PowerTransformer()),
                      ('linear', LinearRegression())])

pl_linear.fit(X_train, y_train)
y_pred = pl_linear.predict(X_test)
print('train mse: ', mean_squared_error(y_train, pl_linear.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))