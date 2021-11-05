# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.tabular import add_datepart


class DataPreprocessing:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

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
        print('------------------------------------Relevent Data------------------------------------')
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())

    def CleanData(self):
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
        """Helper function that adds columns relevant to a date in the column field_name of self.df.
    For example if we have a series of dates we can then generate features such as Year, Month, Day, Dayofweek, Is_month_start, etc as shown below:
    """

        add_datepart(new_data, 'Date')
        new_data.drop('Elapsed', axis=1, inplace=True)

        # Usually stock market is avaliable to buissness days.
        new_data['mon_fri'] = 0
        for i in range(0, len(new_data)):
            if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
                new_data['mon_fri'][i] = 1
            else:
                new_data['mon_fri'][i] = 0
        new_data['Close'] = new_data['Close'].astype(np.float64)
        train = pd.DataFrame(new_data[:1800])
        test = pd.DataFrame(new_data[1800:])
        return train, test

    def Visualize(self):
        """# Data Visualization"""
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%m/%d/%Y')
        self.df.index = self.df['Date']
        plt.figure(figsize=(16, 8))
        plt.plot(self.df['Close'], label='Close Price History')


# using only relevant data


# '''
# train.to_csv('train.csv',index = False)
# test.to_csv('test.csv',index = False)


# Data Preprocessing
