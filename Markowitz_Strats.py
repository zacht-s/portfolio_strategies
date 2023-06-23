import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from datetime import datetime, timedelta
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 100)


def risk(weights, cov_matrix):
    return np.matmul(np.matmul(weights, cov_matrix), weights)


class MarkowitzStrat:
    def __init__(self, tickers, start, end, lookback, max_holding=0.5, long_only=False):
        """
        :param tickers: List of Strings, asset tickers for retrieving price data from Yahoo API
        :param start: datetime object, first trading day
        :param end: datetime object, last trading day
        :param lookback: Int. Number of leading days (trading days) to use to get Exp[Returns] and Covariance Matrix
        :param long_only: Boolean, True if only positive asset weights are allowed.
        """
        self.tickers = tickers
        self.start = start
        self.end = end
        self.lookback = lookback
        self.max_holding = max_holding
        self.long_only = long_only

        # Get Asset Prices / Returns from Yahoo. Note Yfinance call takes calendar days as input
        data_start = start - timedelta(days=self.lookback / 252 * 365)
        self.prices = yf.download(self.tickers, start=data_start, end=self.end)['Adj Close']
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

        # Get Rolling "lookback" length window of asset returns and covariance matrix
        self.__lookback_rets = self.returns.rolling(self.lookback).mean().shift(1).dropna()
        self.__lookback_cov_mtrx = self.returns.rolling(self.lookback).cov().dropna().reset_index()

        # Shift the Multi Index DF of Covariance Matrices back one timestep
        to_drop = self.__lookback_cov_mtrx['Date'][self.__lookback_cov_mtrx.index[-1]]
        new_index1 = self.__lookback_cov_mtrx['Date'][len(self.tickers):]
        new_index2 = self.__lookback_cov_mtrx['level_1'][len(self.tickers):]
        self.__lookback_cov_mtrx = \
            self.__lookback_cov_mtrx.drop(self.__lookback_cov_mtrx.index[self.__lookback_cov_mtrx['Date'] == to_drop])

        self.__lookback_cov_mtrx = self.__lookback_cov_mtrx.set_index([new_index1, new_index2])
        self.__lookback_cov_mtrx = self.__lookback_cov_mtrx.drop(['Date', 'level_1'], axis=1)

    def simulate(self, tgt_return):
        w0 = np.ones(len(self.tickers)) / len(self.tickers)
        daily_tgt = tgt_return / 252

        if self.long_only:
            bound = Bounds(lb=np.zeros(len(self.tickers)), ub=np.zeros(len(self.tickers)) + self.max_holding)
        else:
            bound = Bounds(lb=np.zeros(len(self.tickers)) - self.max_holding,
                           ub=np.zeros(len(self.tickers)) + self.max_holding)

        days = []
        positions = []
        for day in self.__lookback_rets.index[:-1]:
            print(day)
            exp_rets = self.__lookback_rets.loc[day]
            cov_mtrx = self.__lookback_cov_mtrx.loc[day]

            def fully_invested(weights):
                return weights.sum() - 1

            def exp_return(weights):
                return np.matmul(exp_rets.to_numpy(), weights) - daily_tgt

            cons = [{'type': 'eq', 'fun': exp_return},
                    {'type': 'eq', 'fun': fully_invested}]

            result = minimize(risk, w0, method='trust-constr',
                              constraints=cons, bounds=bound,
                              args=(cov_mtrx.to_numpy()), options={'verbose': 0})
            days.append(day)
            positions.append(result.x)

        position_df = pd.DataFrame(positions, index=days, columns=self.__lookback_rets.iloc[0].index)
        print('')
        return position_df


if __name__ == '__main__':

    universe = ['UNH', 'MSFT', 'GS', 'HD', 'MCD', 'CAT', 'AMGN', 'V', 'BA', 'CRM',
                'HON', 'AAPL', 'TRV', 'AXP', 'JNJ', 'CVX', 'WMT', 'PG', 'JPM', 'IBM',
                'NKE', 'MRK', 'MMM', 'DIS', 'KO', 'DOW', 'CSCO', 'VZ', 'INTC', 'WBA']

    test_class = MarkowitzStrat(universe, start=datetime(2023, 1, 1),
                                end=datetime(2023, 6, 15), lookback=252)

    print(test_class.simulate(tgt_return=0.1))

