from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import numpy as np


class LinearRegressor:

    def predict(self, train, test):
        x_train = train.drop('Close', axis=1)
        y_train = train['Close']
        x_test = test.drop('Close', axis=1)
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        return pred

    def Visualize(self, preds, train, test):
        y_test = test['Close']
        rsm = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
        print("Root Mean Square: ",rsm)
        print("R^2 Score: ",r2_score(preds,y_test))
        test['Predictions'] = preds
        plt.figure(figsize=(16, 8))
        plt.plot(train['Close'])
        plt.plot(test[['Close', 'Predictions']])
