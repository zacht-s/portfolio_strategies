import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

'''
Holdings Object(position, pos_type, asset_prices=false)
.maxdrawdown
    .var
    .return
    .risk
    .pnl
    .sharpe
'''


class Holdings:
    def __init__(self, position, pos_type):
        """
        :param position: pandas dataframe of portfolio positions indexed by time, columns are stock tickers
                         The convention is that the portfolio can be bought at the indexed timestep
                         (i.e. was built on lagged data)

        :param pos_type: "Total" or "Delta". Total means that each dataframe entry is the optimal portfolio weight.
                         Delta means that each entry is the change in portfolio from the prior timestep.

        :param val_prices: Optional dataframe of prices to buy/sell portfolio. index=timestamp, column should be tickers
                           which match the columns of the position dataframe.
        """

        if pos_type.lower() == 'total':
            self.gross_position = position
            self.delta_position = position - position.shift(1)
            self.delta_position.iloc[0] = position.iloc[0]
        elif pos_type.lower() == 'delta':
            self.delta_position = position
            self.gross_position = position.cumsum()
        else:
            raise TypeError('pos_type must be either "total" or "delta"')

        start = self.gross_position.index[0]
        end = self.gross_position.index[-1] + timedelta(days=10)
        prices = yf.download(tickers=list(self.gross_position.columns), start=start, end=end)['Adj Close']
        closeout_day = prices.index[prices.index.get_loc(self.gross_position.index[-1])+1]
        self.prices = prices.loc[start:closeout_day]

        self.delta_position.loc[closeout_day] = -self.gross_position.iloc[-1]
        self.gross_position.loc[closeout_day] = np.zeros(len(self.gross_position.columns))

        self.holdings_value = (self.gross_position * self.prices).sum(axis=1)
        self.cashflow = (self.delta_position * self.prices * -1).sum(axis=1)
        self.portfolio_value = self.holdings_value + self.cashflow.cumsum()

    def pnl(self):
        starting_investment = self.holdings_value.iloc[0]
        profit_and_loss = (starting_investment + self.portfolio_value) / starting_investment * 100 - 100
        profit_and_loss = round(profit_and_loss, 3)
        return profit_and_loss

    def buy_and_hold_return(self):
        """
        Calculates the return assuming you just bought the initial portfolio and sold at the end of trading period
        """
        bnh_holdings = pd.DataFrame([self.gross_position.iloc[0], self.gross_position.iloc[0]],
                                    index=[self.prices.index[0], self.prices.index[-1]],
                                    columns=self.prices.columns)
        starting_investment = (bnh_holdings.iloc[0] * self.prices.iloc[0]).sum()
        closeout_value = (bnh_holdings.iloc[-1] * self.prices.iloc[-1]).sum()

        return round((closeout_value - starting_investment) / starting_investment * 100, 3)

    def port_return(self):
        profit_and_loss = self.pnl()
        return profit_and_loss[-1] - profit_and_loss[0]

    def port_risk(self):
        profit_and_loss = self.pnl()
        daily_rets = (profit_and_loss - profit_and_loss.shift(1)).dropna()
        risk = daily_rets.std()
        return round(risk, 2)

    def sharpe_ratio(self):
        rf_rates = yf.download('^IRX', start=self.prices.index[0], end=self.prices.index[-1])['Adj Close']
        rf = rf_rates.mean()

        sharpe = (self.port_return() - rf) / self.port_risk()
        return sharpe

    def max_drawdown(self, days=10):
        profit_and_loss = self.pnl()
        drawdowns = (profit_and_loss - profit_and_loss.shift(days)).dropna()
        return min(drawdowns) * -1


if __name__ == '__main__':
    test_df = pd.read_csv('test_position.csv', index_col='Date')
    test_df.index = pd.to_datetime(test_df.index)
    test_holdings = Holdings(position=test_df, pos_type='Total')

    #print(test_holdings.pnl())
    #print(test_holdings.buy_and_hold_return())
    #print(test_holdings.port_return())
    #print(test_holdings.port_risk())
    #print(test_holdings.sharpe_ratio())
    #print(test_holdings.max_drawdown())

    # Look at adding VAR another time also
