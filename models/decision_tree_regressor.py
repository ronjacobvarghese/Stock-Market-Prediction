from sklearn.tree import DecisionTreeRegressor 
import matplotlib.pyplot as plt


class DecisionTreeRegression:

    def predict(self, train, test):
        test = test.drop('Predictions',axis = 1)
        x_train = train.drop('Close', axis=1)
        y_train = train['Close']
        x_test = test.drop('Close', axis=1)
        regressor = DecisionTreeRegressor(random_state = 0) 
        regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        return pred

    def Visualize(self, preds, train, test):
        test['Predictions'] = preds
        plt.figure(figsize=(16, 8))
        plt.plot(train['Close'])
        plt.plot(test[['Close', 'Predictions']])
