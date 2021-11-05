from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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
        test['Predictions'] = preds
        plt.figure(figsize=(16, 8))
        plt.plot(train['Close'])
        plt.plot(test[['Close', 'Predictions']])
