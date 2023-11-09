import  pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_decomposition, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pylot as plt
from  matplotlib import style

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    style.use('ggplot')

    df = Quandl.get('WIKI/GOOGL') # WIKI/GOOGL is the call from the Quandl which get a dataset
    df = df[['Adj. open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]] # get these parameters
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    df = df[['Adj. Close' , 'HL_PCT' , 'PCT_change', 'Adj. Volume']] #create new table

    print(df.head())

    forcast_col = 'Adj. Close'
    df.fillnae(-9999, inplace = True)
    # fillna is fill non-available : when we don't have this data but do not want to lose the exist data we will let "defult" data to the "NA"

    forcast_out = int(math.ceil( 0.01 * len(df)))
    # get 1% of the rows in this site, it represents the number of days into the future we want predict

    df['label'] = df[forcast_col].shift(-forcast_out)
    # set the value into a new column

    df.dropna(inplace = True)
    #don't show the NaN

    X = np.array(df.drop(['label'] ,1))
    # into X we get all the data except 'label' - it's also a data frame

    X = preprocessing.scale(X)
    # feed himself with new values while it is changing in realtime
    # scale the value by the new value

    X = X[:-forcast_out] # the current data
    X_lately = X[-forcast_out:]
    # x_lately is the stuff we will use to predict again - currently we don't have this values

    Y = np.array(df['label'])  # into Y we get only the column 'label'
    Y = np.array(df['label'])

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)
    # we take 20% from the date to use as testing data
    # "shuffle the data where is the best accuracy to output it to the classifier

    clf = LinearRegression(n_jobs = 4)
    # THIS IS THE LEARNING ALGORITHM - we can change LinearRegression() to svm.SVR() etc
    # svm is "support vector machine" learning and in it you can fill as svm.SVR(kernel = 'poly'/'linear/etc)
    # n_job is attribute of svm that allocate how many threads will run. (-1) is for many threads as can

    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test) # the accuracy which is the square Error: (y^(i) - θ^(t) * X^(i))^2
    # we use part of the info we know to use it as test to the prediction of the machine
    # x_train, y_train is the model the machine learn from and predict
    # x_test, y_test is the model we want to check the accuracy
    #print(accuracy) will print the accuracy of the true value

    forcast_set = clf.predict(X_lately)
    # the forcast the using the classifier which predict by the value we get earlier
    # print(forcast_set) will print the

    df['Forcast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400 # second at day
    next_unix = last_unix + one_day
    # we need to organize our features and wew want to add him the date that we can control the data

    # we going to add the date stamp into every day
    for i in forcast_set:
        next_date = datetime.datetime.fromtimestamp((next_unix))
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] +[i] # is a list of nan unless the is a value
        #df.loc[next_date] is like the index of the dataFrame

    # represnt the graph with the predicted values
    df['Adj. Close'].plot()
    df["Forcast"].plot()
    plt.legend(loc = 4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
