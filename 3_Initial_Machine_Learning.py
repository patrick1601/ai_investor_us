from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from pandas.plotting import scatter_matrix
#%%
# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150
#%%
def loadXandyAgain():
    '''
    Load X and y.
    Randomises rows.
    Returns X, y.
    '''
    # Read in data
    X = pd.read_csv("Annual_Stock_Price_Fundamentals_Ratios.csv",
                    index_col=0)
    y = pd.read_csv("Annual_Stock_Price_Performance_Percentage.csv",
                    index_col=0)
    y = y["Perf"]  # We only need the % returns as target

    # randomize the rows
    X['y'] = y
    X = X.sample(frac=1.0, random_state=42)  # randomize the rows
    y = X['y']
    X.drop(columns=['y'], inplace=True)

    return X, y
#%%
X, y = loadXandyAgain()
print(y.mean()) # Average stock return if we were picking at random.
#%% LINEAR REGRESSION
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)
#%%
# Try out linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

pl_linear = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('linear', LinearRegression())
    ])

pl_linear.fit(X_train, y_train)
y_pred = pl_linear.predict(X_test)

print('Train MSE: ',
      mean_squared_error(y_train, pl_linear.predict(X_train)))
print('Test MSE: ',
      mean_squared_error(y_test, y_pred))

import pickle # To save the fitted model
pickle.dump(pl_linear, open("pl_linear.p", "wb" ))
#%% try many runs to see mean squared error with different training set sizes
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

sizesToTrain = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]
train_sizes, train_scores, test_scores, fit_times, score_times = \
    learning_curve(pl_linear, X, y, cv=ShuffleSplit(n_splits=100,
                                                    test_size=0.2,
                                                    random_state=42),
                   scoring='neg_mean_squared_error',
                   n_jobs=4, train_sizes=sizesToTrain,
                   return_times=True)

results_df = pd.DataFrame(index=train_sizes) #Create a DataFrame of results
results_df['train_scores_mean'] = np.sqrt(-np.mean(train_scores, axis=1))
results_df['train_scores_std'] = np.std(np.sqrt(-train_scores), axis=1)
results_df['test_scores_mean'] = np.sqrt(-np.mean(test_scores, axis=1))
results_df['test_scores_std'] = np.std(np.sqrt(-test_scores), axis=1)
results_df['fit_times_mean'] = np.mean(fit_times, axis=1)
results_df['fit_times_std'] = np.std(fit_times, axis=1)
#%% plot learning curve for linear regression
results_df['train_scores_mean'].plot(style='-x')
results_df['test_scores_mean'].plot(style='-x')

plt.fill_between(results_df.index,\
                 results_df['train_scores_mean']-results_df['train_scores_std'],\
                 results_df['train_scores_mean']+results_df['train_scores_std'], alpha=0.2)
plt.fill_between(results_df.index,\
                 results_df['test_scores_mean']-results_df['test_scores_std'],\
                 results_df['test_scores_mean']+results_df['test_scores_std'], alpha=0.2)
plt.grid()
plt.legend(['Training CV RMSE Score','Test Set RMSE'])
plt.ylabel('RMSE')
plt.xlabel('Number of training rows')
plt.title('Linear Regression Learning Curve', fontsize=15)
#plt.ylim([0, 1.25]);
plt.show()
#%% function to plot contour plot of predicted vs. actual
# Output scatter plot and contour plot of density of points to see if prediciton matches reality
# Line of x=y is provided, perfect prediction would have all density on this line
# Also plot linear regression of the scatter

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plotDensityContourPredVsReal(model_name, x_plot, y_plot, ps):
    # Plotting scatter
    plt.scatter(x_plot, y_plot, s=1)
    # Plotting linear regression
    # Swap X and Y fit because prediction is quite centered around one value.
    LinMod = LinearRegression().fit(y_plot.reshape(-1, 1), x_plot.reshape(-1, 1))
    xx=[[-5],[5]]
    yy=LinMod.predict(xx)
    plt.plot(yy,xx,'g')
    # Plot formatting
    plt.grid()
    plt.axhline(y=0, color='r', label='_nolegend_')
    plt.axvline(x=0, color='r', label='_nolegend_')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.plot([-100,100],[-100,100],'y--')
    plt.xlim([-ps,ps])
    plt.ylim([-ps,ps])
    plt.title('Predicted/Actual density plot for {}'.format(model_name))
    plt.legend(['Linear Fit Line','y=x Perfect Prediction Line','Prediction Points'])
    # Save Figure
    #plt.figure(figsize=(5,5))
    plt.savefig('result.png')
#%% plot figure
plt.figure(figsize=(6,6))
plotDensityContourPredVsReal('pl_linear', y_pred, y_test.to_numpy(), 2)
#%% compare the top 10 stocks the model chooses to actual performance
# See top 10 stocks and see how the values differ
# Put results in a DataFrame so we can sort it.
y_results = pd.DataFrame()
y_results['Actual Return'] = y_test
y_results['Predicted Return'] = y_pred

# Sort it by the predicted return.
y_results.sort_values(by='Predicted Return',
                      ascending=False,
                      inplace=True)
y_results.reset_index(drop=True,
                      inplace=True)


print('Predicted Returns:', list(np.round(y_results['Predicted Return'].iloc[:10],2)))
print('Actual Returns:', list(np.round(y_results['Actual Return'].iloc[:10],2)))
print('\nTop 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[:10].mean(),2) , '%','\n')
print('Actual Top 10 Returns:', round(y_results['Actual Return'].iloc[:10].mean(),2) , '%','\n')
#%% try the bottom 10
# See bottom 10 stocks and see how the values differ
print('\nBottom 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[-10:].mean(),2) , '%','\n')
print('Actual Bottom 10 Returns:', round(y_results['Actual Return'].iloc[-10:].mean(),2) , '%','\n')
#%% function to try a few top10/bottom10 runs by changing the random state of the test/train split
def observePredictionAbility(my_pipeline, X, y, returnSomething=False, verbose=True):
    '''
    For a given predictor pipeline.
    Create table of top10/bottom 10 averaged,
    10 rows of 10 random_states.
    to give us a synthetic performance result.
    Prints Top and Bottom stock picks

    The arguments returnSomething=False, verbose=True,
    will be used at the notebook end to get results.
    '''
    Top10PredRtrns, Top10ActRtrns = [], []
    Bottom10PredRtrns, Bottom10ActRtrns = [], []

    for i in range(0, 10):  # Can try 100
        # Pipeline and train/test
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.1, random_state=42 + i)
        my_pipeline.fit(X_train, y_train)
        y_pred = my_pipeline.predict(X_test)

        # Put results in a DataFrame so we can sort it.
        y_results = pd.DataFrame()
        y_results['Actual Return'] = y_test
        y_results['Predicted Return'] = y_pred

        # Sort it by the prediced return.
        y_results.sort_values(by='Predicted Return',
                              ascending=False,
                              inplace=True)
        y_results.reset_index(drop=True,
                              inplace=True)

        # See top 10 stocks and see how the values differ
        Top10PredRtrns.append(
            round(np.mean(y_results['Predicted Return'].iloc[:10]) * 100,
                  2))
        Top10ActRtrns.append(
            round(np.mean(y_results['Actual Return'].iloc[:10]) * 100,
                  2))

        # See bottom 10 stocks and see how the values differ
        Bottom10PredRtrns.append(
            round(np.mean(y_results['Predicted Return'].iloc[-10:]) * 100,
                  2))
        Bottom10ActRtrns.append(
            round(np.mean(y_results['Actual Return'].iloc[-10:]) * 100,
                  2))

    if verbose:
        print('Predicted Performance of Top 10 Return Portfolios:',
              Top10PredRtrns)
        print('Actual Performance of Top 10 Return Portfolios:',
              Top10ActRtrns, '\n')
        print('Predicted Performance of Bottom 10 Return Portfolios:',
              Bottom10PredRtrns)
        print('Actual Performance of Bottom 10 Return Portfolios:',
              Bottom10ActRtrns)
        print('--------------\n')

        print('Mean Predicted Std. Dev. of Top 10 Return Portfolios:',
              round(np.array(Top10PredRtrns).std(), 2))
        print('Mean Actual Std. Dev. of Top 10 Return Portfolios:',
              round(np.array(Top10ActRtrns).std(), 2))
        print('Mean Predicted Std. Dev. of Bottom 10 Return Portfolios:',
              round(np.array(Bottom10PredRtrns).std(), 2))
        print('Mean Actual Std. Dev. of Bottom 10 Return Portfolios:',
              round(np.array(Bottom10ActRtrns).std(), 2))
        print('--------------\n')

        # PERFORMANCE MEASURES HERE
        print( \
            '\033[4mMean Predicted Performance of Top 10 Return Portfolios:\033[0m', \
            round(np.mean(Top10PredRtrns), 2))
        print( \
            '\t\033[4mMean Actual Performance of Top 10 Return Portfolios:\033[0m', \
            round(np.mean(Top10ActRtrns), 2))
        print('Mean Predicted Performance of Bottom 10 Return Portfolios:', \
              round(np.mean(Bottom10PredRtrns), 2))
        print('\tMean Actual Performance of Bottom 10 Return Portfolios:', \
              round(np.mean(Bottom10ActRtrns), 2))
        print('--------------\n')

    if returnSomething:
        # Return the top10 and bottom 10 predicted stock return portfolios
        # (the actual performance)
        return Top10ActRtrns, Bottom10ActRtrns

    pass
#%%
observePredictionAbility(pl_linear, X, y)
#%%ELASTIC NET REGRESSION
# ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PowerTransformer

pl_ElasticNet = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('ElasticNet', ElasticNet())#l1_ratio=0.00001, alpha=0.001
])

pl_ElasticNet.fit(X_train, y_train)
y_pred = pl_ElasticNet.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_ElasticNet.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, pl_ElasticNet.predict(X_test)))

import pickle
pickle.dump(pl_ElasticNet, open("pl_ElasticNet.p", "wb" ))
#%% plot contour plot
plt.figure(figsize=(5,5))
plotDensityContourPredVsReal('pl_ElasticNet', y_pred, y_test.to_numpy(),2)
#%% test some hyper parameters to improve prediction
# ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PowerTransformer

pl_ElasticNet = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('ElasticNet', ElasticNet(l1_ratio=0.00001))
])

pl_ElasticNet.fit(X_train, y_train)
y_pred_lowL1 = pl_ElasticNet.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_ElasticNet.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, pl_ElasticNet.predict(X_test)))

#import pickle
#pickle.dump(pl_ElasticNet, open("pl_ElasticNet.p", "wb" ))
#%% plot contour plot again
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plotDensityContourPredVsReal('pl_ElasticNet', y_pred, y_test.to_numpy(),2)
plt.title('Elasticnet Default Hyperparameters',fontsize=15)
plt.subplot(1,2,2)
plotDensityContourPredVsReal('pl_ElasticNet', y_pred_lowL1, y_test.to_numpy(),2)
plt.title('Elasticnet L1 Ratio=0.00001',fontsize=15)
plt.show()
#%%
def PrintTopAndBottom10Predictions(y_test, y_pred):
    '''
    See top 10 stocks and see how the values differ.
    Returns nothing.
    '''

    # Put results in a DataFrame so we can sort it.
    y_results = pd.DataFrame()
    y_results['Actual Return'] = y_test
    y_results['Predicted Return'] = y_pred

    # Sort it by the prediced return.
    y_results.sort_values(by='Predicted Return',
                          ascending=False,
                          inplace=True)
    y_results.reset_index(drop=True,
                          inplace=True)

    # print('Predicted Returns:', list(np.round(y_results['Predicted Return'].iloc[:10],2)))
    # print('Actual Returns:', list(np.round(y_results['Actual Return'].iloc[:10],2)))
    # Top 10
    print('\nTop 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[:10].mean(), 2), '%', '\n')
    print('Actual Top 10 Returns:', round(y_results['Actual Return'].iloc[:10].mean(), 2), '%', '\n')
    # Bottom 10
    print('\nBottom 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[-10:].mean(), 2), '%', '\n')
    print('Actual Bottom 10 Returns:', round(y_results['Actual Return'].iloc[-10:].mean(), 2), '%', '\n')

    pass
#%%
PrintTopAndBottom10Predictions(y_test, y_pred)
#%% see if the results are repeatabile
observePredictionAbility(pl_ElasticNet, X, y)
#%% K NEAREST NEIGHBOURS REGRESSION
# Read in data and do train/test split.
X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)
#%%
# KNeighbors regressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PowerTransformer
pl_KNeighbors = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=40))
])

pl_KNeighbors.fit(X_train, y_train)
y_pred = pl_KNeighbors.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_KNeighbors.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))

import pickle
pickle.dump(pl_KNeighbors, open("pl_KNeighbors.p", "wb" ))
#%% check validation curve
knn_validation_list = []
numNeighbours = [4, 8, 16, 32, 64, 100]
runNum = 40

for i in numNeighbours:
    print('Trying K=' + str(i))
    for j in range(0, runNum):
        # Get a new train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            random_state=42 + j)
        pl_KNeighbors = Pipeline([
            ('Power Transformer', PowerTransformer()),
            ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=i))
        ]).fit(X_train, y_train)

        y_pred = pl_KNeighbors.predict(X_test)

        resultThisRun = [i,
                         mean_squared_error(y_train, pl_KNeighbors.predict(X_train)),
                         mean_squared_error(y_test, y_pred)]

        knn_validation_list.append(resultThisRun)

knn_validation_df = pd.DataFrame(knn_validation_list,
                                 columns=['numNeighbours',
                                          'trainError',
                                          'testError'])
#%% create dataframe to plot results
# Get our results in a format we can easily plot
knn_results_list = []
numNeighboursAttempted = knn_validation_df['numNeighbours'].unique()
results_df = pd.DataFrame(index=numNeighboursAttempted)  # Create a DataFrame of results

for i in numNeighboursAttempted:
    blNeighbours = knn_validation_df['numNeighbours'] == i  # boolean mask

    trainErrorsMean = knn_validation_df[blNeighbours]['trainError'].mean()
    trainErrorsStd = knn_validation_df[blNeighbours]['trainError'].std()
    testErrorsMean = knn_validation_df[blNeighbours]['testError'].mean()
    testErrorsStd = knn_validation_df[blNeighbours]['testError'].std()
    knn_results_list.append([trainErrorsMean, trainErrorsStd,
                             testErrorsMean, testErrorsStd])

knn_results_df = pd.DataFrame(knn_results_list,
                              columns=['trainErrorsMean', 'trainErrorsStd',
                                       'testErrorsMean', 'testErrorsStd'],
                              index=numNeighboursAttempted)
#%%
knn_results_df['trainErrorsMean'].plot(style='-x', figsize=(7,4.5))
knn_results_df['testErrorsMean'].plot(style='-x')

plt.fill_between(knn_results_df.index,\
                 knn_results_df['trainErrorsMean']
                 -knn_results_df['trainErrorsStd'],
                 knn_results_df['trainErrorsMean']
                 +knn_results_df['trainErrorsStd'], alpha=0.2)
plt.fill_between(results_df.index,\
                 knn_results_df['testErrorsMean']
                 -knn_results_df['testErrorsStd'],
                 knn_results_df['testErrorsMean']
                 +knn_results_df['testErrorsStd'], alpha=0.2)
plt.grid()
plt.legend(['Training CV MSE Score','Test Set MSE'])
plt.ylabel('MSE', fontsize=15)
plt.xlabel('Values for K', fontsize=15)
plt.title('K-NN Validation Curve', fontsize=20)
#plt.ylim([0, 0.8]);
plt.show()
#%% plot the scatter graph
plt.figure(figsize=(6,6))
plotDensityContourPredVsReal('pl_KNeighbors', y_pred, y_test.to_numpy(),2)
#%% See top 10 stocks and see how the values differ
PrintTopAndBottom10Predictions(y_test, y_pred)
#%%
observePredictionAbility(pl_KNeighbors, X, y)
#%% KDE Plot
# Output scatter plot and contour plot of density of points to see if prediciton matches reality
# Line of x=y is provided, perfect prediction would have all density on this line
# Also plot linear regression of the scatter

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde # Need for Kernel Density Calculation
from sklearn.linear_model import LinearRegression

def plotDensityContourPredVsRealContour(model_name, x_plot, y_plot, ps):
#x_plot, y_plot = y_pred, y_test.to_numpy()
    resolution = 40
    # Make a gaussian kde on a grid
    k = kde.gaussian_kde(np.stack([x_plot,y_plot]))
    xi, yi = np.mgrid[-ps:ps:resolution*4j, -ps:ps:resolution*4j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # Plotting scatter and contour plot
    plt.pcolormesh(xi,
                   yi,
                   zi.reshape(xi.shape),
                   shading='gouraud',
                   cmap=plt.cm.Greens)
    plt.contour(xi, yi, zi.reshape(xi.shape) )
    plt.scatter(x_plot, y_plot, s=1)
    # Plotting linear regression
    LinMod = LinearRegression()
    LinMod.fit(y_plot.reshape(-1, 1), x_plot.reshape(-1, 1),)
    xx=[[-2],[2]]
    yy=LinMod.predict(xx)
    plt.plot(yy,xx)
    # Plot formatting
    plt.grid()
    plt.axhline(y=0, color='r', label='_nolegend_')
    plt.axvline(x=0, color='r', label='_nolegend_')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.plot([-100,100],[-100,100],'y--')
    plt.xlim([-ps*0.2,ps*0.6]) #
    plt.ylim([-ps,ps])
    plt.title('Predicted/Actual density plot for {}'.format(model_name))
    plt.legend([
        'Linear Fit Line','y=x Perfect Prediction Line',
        'Prediction Points'])
    # Save Figure
    plt.savefig('result.png')
#%%
plt.figure(figsize=(7,7))
plotDensityContourPredVsRealContour('pl_KNeighbors', y_pred, y_test.to_numpy(),1)
#%% SUPPORT VECTOR MACHINE REGRESSION
# Read in data and do train/test split.
X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)
#%%
# SVM quick and dirty
from sklearn.svm import SVR

pl_svm = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('SVR', SVR()) # kernel='rbf', C=100, gamma=0.1, epsilon=.1 generated good returns.
])

pl_svm.fit(X_train, y_train)
y_pred = pl_svm.predict(X_test)
from sklearn.metrics import mean_squared_error
print('mse: ', mean_squared_error(y_test, y_pred))
from sklearn.metrics import mean_absolute_error
print('mae: ', mean_absolute_error(y_test, y_pred))

import pickle
pickle.dump(pl_svm, open("pl_svm.p", "wb" ))
#%%
observePredictionAbility(pl_svm, X, y)
#%As there are many parameters do a GridSearch CV, to find optimal parameters
# Takes a LONG time
from sklearn.model_selection import GridSearchCV

parameters = [{'SVR__kernel': ['linear'],\
               'SVR__C': [1, 10, 100],\
               'SVR__epsilon': [0.05, 0.1, 0.2]},\
                {'SVR__kernel': ['rbf'],\
               'SVR__C': [1, 10, 100],\
               'SVR__gamma': [0.001, 0.01, 0.1],\
               'SVR__epsilon': [0.05, 0.1, 0.2]} ] # Best found was C=1, gamma=0.001, epsilon=0.2

svm_gs = GridSearchCV(pl_svm, parameters, cv=10, scoring='neg_mean_squared_error')

svm_gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(svm_gs.best_params_,'\n')
print("Grid scores on development set:")
means = svm_gs.cv_results_['mean_test_score']
stds = svm_gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))