import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SvrRegressor:

    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.sc = StandardScaler()

    def fit_standize(self, x):
        x = self.sc.fit_transform(x)
        return x

    def standize(self, x):
        x = self.sc.transform(x)
        return x

    def predict(self):
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']
        x_test = self.test.drop('Close', axis=1)
        x_train = self.fit_standize(x_train)
        x_test = self.standize(x_test)
        regressor = SVR(kernel='rbf')
        regressor.fit(x_train, y_train)
        pred = regressor(x_test)
        return pred

    def Visualize(self, preds):
        self.test['Predictions'] = preds
        plt.figure(figsize=(16, 8))
        plt.plot(self.train['Close'])
        plt.plot(self.test[['Close', 'Predictions']])
