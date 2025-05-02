import gym
from gym import spaces
import numpy as np
import pandas as pd

class MultiStockTradingEnv(gym.Env):
    def __init__(self, df, ticker):
        super(MultiStockTradingEnv, self).__init__()
        self.df = df[df['Ticker'] == ticker].reset_index(drop=True)
        self.ticker = ticker
        self.initial_balance = 10000
        self.reset()

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: 7 indicators + Close + balance + position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def _get_obs(self):
        row = self.df.loc[self.current_step]
        obs = np.array([
            row['SMA_20'], row['EMA_20'], row['RSI_14'],
            row['MACD'], row['MACD_signal'],
            row['BB_upper'], row['BB_lower'],
            row['Close'], self.balance, self.position
        ], dtype=np.float32)
        return obs

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.net_worth = self.initial_balance
        return self._get_obs()

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        # Buy
        if action == 1 and self.balance >= current_price:
            self.position += 1
            self.balance -= current_price

        # Sell
        elif action == 2 and self.position > 0:
            self.position -= 1
            self.balance += current_price

        # Advance one step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Recalculate net worth
        self.net_worth = self.balance + self.position * current_price
        reward = self.net_worth - self.initial_balance

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"[{self.ticker}] Step: {self.current_step} | "
              f"Balance: {self.balance:.2f} | Position: {self.position} | "
              f"Net Worth: {self.net_worth:.2f}")
