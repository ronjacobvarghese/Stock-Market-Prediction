import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


class Lstm:

    def __init__(self, filename):
        self.file = filename
        self.df = pd.read_csv(filename)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = Sequential()

    def ProcessData(self):
        self.df['Close'] = self.df[' Close/Last']
        self.df = self.df.drop(' Close/Last', axis=1)
        self.df['Volume'] = self.df[' Volume']
        self.df = self.df.drop(' Volume', axis=1)
        self.df['Open'] = self.df[' Open']
        self.df = self.df.drop(' Open', axis=1)
        self.df['High'] = self.df[' High']
        self.df = self.df.drop(' High', axis=1)
        self.df['Low'] = self.df[' Low']
        self.df = self.df.drop(' Low', axis=1)

        for i in range(len(self.df)):
            self.df['Close'][i] = self.df['Close'][i][2:]
            self.df['Open'][i] = self.df['Open'][i][2:]
            self.df['High'][i] = self.df['High'][i][2:]
            self.df['Low'][i] = self.df['Low'][i][2:]

        self.df["Close"] = self.df['Close'].astype(np.float64)
        self.df["Open"] = self.df['Open'].astype(np.float64)
        self.df["High"] = self.df['High'].astype(np.float64)
        self.df["Low"] = self.df['Low'].astype(np.float64)

    def CleanData(self):
        self.ProcessData()
        self.df.index = self.df['Date']
        data = self.df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(
            0, len(self.df)), columns=['Date', 'Close'])
        for i in range(0, len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Close'][i] = data['Close'][i]
        new_data['Date'] = pd.to_datetime(new_data['Date'], format='%m/%d/%Y')
        new_data.index = new_data['Date']
        new_data = new_data.sort_index(ascending=True, axis=0)
        new_data.drop('Date', axis=1, inplace=True)
        dataset = new_data.values
        return new_data, dataset

    def Normalize(self, dataset):
        scaled_data = self.scaler.fit_transform(dataset)
        return scaled_data

    def PrepData(self, dataset):
        dataset = self.Normalize(dataset)
        x_train, y_train = [], []
        for i in range(60, 1800):
            x_train.append(dataset[i-60:i, 0])
            y_train.append(dataset[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        return x_train, y_train

    def fit_Model(self, dataset):
        x_train, y_train = self.PrepData(dataset)
        self.model.add(LSTM(units=50, return_sequences=True,
                       input_shape=(x_train.shape[1], 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    def PrepTest(self, dataset):
        inputs = dataset[1800 - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)
        x_test = []
        for i in range(60, inputs.shape[0]):
            x_test.append(inputs[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test

    def predict(self, dataset):
        self.fit_Model(dataset)
        x_test = self.PrepTest(dataset)
        closing_price = self.model.predict(x_test)
        closing_price = self.scaler.inverse_transform(closing_price)
        return closing_price

    def Visualize(self, dataset, preds):
        train = dataset[:1800]
        valid = dataset[1800:]
        plt.figure(figsize=(16, 8))
        valid['Predictions'] = preds
        plt.plot(train['Close'], label='Training')
        plt.plot(valid[['Close']], label='Valid')
        plt.plot(valid[['Predictions']], label='Predicted')
        plt.legend()
