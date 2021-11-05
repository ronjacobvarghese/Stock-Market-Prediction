import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LinearScratcher:

    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.m = len(train)
        self.encoder = LabelEncoder()

    def convert(self, dataset, column):
        dataset[column] = self.encoder.fit_transform(dataset[column])

    def cost_function(self, x, y, theta):
        y_pred = np.dot(x, theta)
        error = (y_pred-y)**2
        cost = 1/(2*self.m)*np.sum(error)
        return cost

    def gradient_descent(self, x, y, theta, alpha, epochs=2000):
        costs = []
        for i in range(0, epochs):
            y_pred = np.dot(x, theta)
            D = np.dot(x.transpose(), (y_pred-y))/self.m
            theta -= alpha*D
            costs.append(self.cost_function(x, y, theta))
        return costs, theta

    def PrepData(self):
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']
        self.convert(x_train, 'Is_month_end')
        self.convert(x_train, 'Is_month_start')
        self.convert(x_train, 'Is_quarter_end')
        self.convert(x_train, 'Is_quarter_start')
        self.convert(x_train, 'Is_year_end')
        self.convert(x_train, 'Is_year_start')
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_train = y_train.reshape((len(y_train), 1))
        x_train = np.append(np.ones((len(x_train), 1)), x_train, axis=1)
        return x_train, y_train

    def PrepTest(self):
        x_test = self.test.drop('Close', axis=1)
        self.convert(x_test, 'Is_month_end')
        self.convert(x_test, 'Is_month_start')
        self.convert(x_test, 'Is_quarter_end')
        self.convert(x_test, 'Is_quarter_start')
        self.convert(x_test, 'Is_year_end')
        self.convert(x_test, 'Is_year_start')
        return x_test

    def predict(self, theta):
        x_test = self.PrepTest()
        y_pred = np.dot(
            np.append(np.ones((len(x_test), 1)), x_test, axis=1), theta)
        return y_pred

    def Visualize(self, preds):
        self.test['Predictions'] = preds
        plt.figure(figsize=(16, 8))
        plt.plot(self.train['Close'])
        plt.plot(self.test[['Close', 'Predictions']])
