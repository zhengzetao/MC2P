import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
import heapq
import math as mh
from gym import spaces
import matplotlib
import ipdb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from tools.qpsolver import qp_solver

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        indexing,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        lookback,
        tech_indicator_list,
        turbulence_threshold=None,
        # lookback=252,
        day=0,
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.lookback = lookback
        self.df = df
        self.indexing = indexing
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        # self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        # self.agent_num = agent_num

        # action_space normalization and shape is self.stock_dim
        # self.action_space = spaces.Box(low=0, high=self.stock_dim, shape=(self.action_space,))
        self.action_space = spaces.Discrete(self.action_space)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, self.stock_dim)
        )


        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.state = np.array(self.data['price_list'].values[0])
        # self.index_value = np.array(self.data['index_list'].values[0])
        self.index_value = self.state[:,0]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount
        self.tracking_error = [0]

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # asset growth ratio: new_portfolio_value / initial_amount
        self.asset_ratio = [1]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.truth_portfolio_memory = [0]
        self.asset_memory = [range(10)]

        self.date_memory = [self.data.Week_ID.values[0]]

    def step(self, actions):

        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]
            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))
            # print("tracking error:{}".format(self.tracking_error[:10]))
            print("tracking_error:{}".format(np.linalg.norm(self.tracking_error)/len(self.tracking_error)))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # weights = self.softmax_normalization(actions)
            # self.actions_memory.append(weights)
            last_day_memory = self.data
            # calculate the tracking error using qp
            # qp_input = pd.DataFrame(self.state[:,actions]).pct_change().dropna()
            # index_value = pd.DataFrame(self.index_value).pct_change().dropna()
            qp_input = pd.DataFrame(self.state[:,actions+1])
            index_value = pd.DataFrame(self.index_value)
            name_list = ['stock'+str(i) for i in range(len(actions))]
            qp_input.columns = name_list
            # qp_input['index'] = self.index_value
            qp_input['index'] = index_value
            qp = qp_solver(qp_input)
            sol = qp.solve()
            # print('Exact solution with all time-series')
            # print(sol['x'])
            # exit()
            weights = np.zeros(self.stock_dim)
            for k, v in enumerate(actions): weights[v] = sol['x'][k]
            # self.actions_memory.append(np.int64(weights>0))
            self.actions_memory.append(actions)
            # print(actions, sol['x'],weights)
            
            # calculate portfolio return
            # individual stocks' return * weight
            Week_ID = self.data.Week_ID.values[0]
            if self.stock_dim == 430: # for SP500 dataset
                portfolio_return =  np.matmul(self.data.loc[self.day].return_list.values[0], weights)
                truth_indexing = self.indexing.loc[self.day].index_list
            else:
                portfolio_return = sum(
                    self.data['return_list'].values[0][Week_ID,:] * weights
                )
                truth_indexing = self.indexing.values[Week_ID]

            # update portfolio value
            # print(self.data.loc[self.day].Date.values[0])
            # exit()
            self.daily_return_memory.append(portfolio_return)
            new_portfolio_value = self.portfolio_value * (1 + np.sum(portfolio_return))
            self.portfolio_value = new_portfolio_value if new_portfolio_value > 0 else 0

            # save into memory
            self.portfolio_return_memory.append(np.sum(portfolio_return))
            self.truth_portfolio_memory.append(np.sum(truth_indexing))
            
            self.asset_memory.append(new_portfolio_value)
            # self.asset_ratio.append(round(new_portfolio_value/self.asset_memory[-2],5))
            
            if self.stock_dim == 430:
                self.tracking_error.extend((truth_indexing - portfolio_return).tolist())
            else:
                self.tracking_error.append(truth_indexing - portfolio_return) 

            # self.tracking_error = [truth_indexing - portfolio_return]
            # jacarrd similarity
            sim = len(set(self.actions_memory[-1]).intersection(set(self.actions_memory[-2]))) / \
                  len(set(self.actions_memory[-1]).union(set(self.actions_memory[-2])))
            gap = np.mean(abs(truth_indexing - portfolio_return)) * 10
            self.reward = (1 - np.clip(gap,0,1)**(2/5))**(5/2) #- (1 - np.clip(sim,0,1)**(4/5))**(5/4)

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = np.array(self.data['price_list'].values[0])
            # self.index_value = np.array(self.data['index_list'].values[0])
            self.index_value = self.state[:,0]
            self.date_memory.append(self.data.Week_ID.values[0])

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.state = np.array(self.data['price_list'].values[0])
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.truth_portfolio_memory = [0]
        # self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.actions_memory = [range(10)]
        self.daily_return_memory = [range(10)]
        self.date_memory = [self.data.Week_ID.values[0]]
        self.tracking_error = [0]
        return self.state

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        # cpy_action = deepcopy(actions)
        # max_number = heapq.nlargest(10, cpy_action) 
        # max_index = []
        # for t in max_number:
        #     index = cpy_action.index(t)
        #     max_index.append(index)
        #     cpy_action[index] = 0
        # actions = [0 if i not in max_index else actions[i] for i in len(actions)]
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_truth_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.truth_portfolio_memory
        df_truth_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_truth_value

    def save_tracking_error_memory(self):
        date_list = self.date_memory
        tracking_error = self.tracking_error
        df_error_value = pd.DataFrame(
            {"date": date_list, "daily_return": tracking_error}
        )
        return df_error_value

    def save_daily_return_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]
        return_list = self.daily_return_memory
        # df_days_return = pd.DataFrame(
        #     {"date":df_date,"daily_return":return_list}
        #     )
        # print(df_days_return)
        # exit()
        df_days_return = pd.DataFrame(return_list)
        # df_days_return.index = df_date.date
        return df_days_return

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        # df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        state = e.reset()
        return e, state
