import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
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
#%% k-fold cross validation
# try a few values of k
vals = [2,5,10,50,100,200]

for i in vals:
    scores = cross_validate(pl_linear, x, y, scoring='neg_mean_squared_error', cv=i, return_train_score=True)
    print('K=', i, ' Segments')

    # print ('Train scores', np.sqrt(-scores['train_score']))
    # print ('Test scores', np.sqrt(-scores['test_score']))

    print('AVERAGE TEST SCORE:', round(np.sqrt(-scores['test_score']).mean(),4),
          'STD. DEV:', round(np.sqrt(-scores['test_score']).std(),4))

    print('AVERAGE TRAIN SCORE:', round(np.sqrt(-scores['train_score']).mean(),4),
          'STD. DEV:', round(np.sqrt(-scores['train_score']).std(),4))

    print('--------------------------')
#%% function for predicted vs actual
# Output scatter plot and contour plot of density of points to see if prediction matches reality
# Line of x=y is provided to show a perfect prediction
# Also plot linear regression line of best fit

def plotDensityContourPredVsReal(model_name, x_plot, y_plot, ps):
    #plotting scatter
    plt.scatter(x_plot, y_plot, s=1)
    # plotting linear regression
    # swap x and y fit because prediction is quite centered around one value
    LinMod = LinearRegression().fit(y_plot.reshape(-1,1), x_plot.reshape(-1,1))
    xx = [[-5],[5]]
    yy = LinMod.predict(xx)
    plt.plot(yy,xx, 'g')

    # plot formatting
    plt.grid()
    plt.axhline(y=0, color='r', label='_nolegend_')
    plt.axvline(x=0, color='r', label='_nolegend_')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.plot([-100,100], [-100,100], 'y--')
    plt.xlim([-ps,ps])
    plt.ylim([-ps,ps])
    plt.title('Predicted/Actual density plot for {}'.format(model_name))
    plt.legend(['Linear Fit Line', 'y=x Perfect Prediction Line', 'Prediction Points'])
#%% plot predicted vs actual for linear regression model
plotDensityContourPredVsReal('pl_linear', y_pred, y_test.to_numpy(), 2)
plt.show()
#%% check the top 10 predicted stocks and see how they differ to actual
y_predtest = pd.DataFrame(y_pred)
bl_top10 = (y_predtest[0] > y_predtest.nlargest(10,0).tail(1)[0].values[0]) # top 10 predicted returns
y_test_reindexed = y_test.reset_index(drop=True)

print('Predicted Returns:', y_predtest[bl_top10][0].values)
print('\nActual Returns:', y_test_reindexed[bl_top10].values)

# combine the top 10 stocks in a portfolio and check performance
print('\nTop 10 Predicted Returns:', round(np.mean(y_predtest[bl_top10][0])*100,2), '%')
print('Actual Top 10 Returns:', round(np.mean(y_test_reindexed[bl_top10])*100,2), '%', '\n')

# combine the top 10 stocks in a portfolio and check performance
y_predtest = pd.DataFrame(y_pred)
bl_bottom10 = (y_predtest[0] < y_predtest.nsmallest(8,0).tail(1)[0].values[0]) # bottom 10 predicted returns
print('Bottom 10 Predicted Returns:', round(np.mean(y_predtest[bl_bottom10][0])*100,2), '%')
print('Actual Bottom 10 Returns:', round(np.mean(y_test_reindexed[bl_bottom10])*100,2), '%')