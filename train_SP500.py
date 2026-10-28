import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime
import shutil
import os

import config
from preprocessor.dataloader import DataLoader
from preprocessor.preprocessors import FeatureEngineer, data_split_SP500
# from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
from env.env_portfolio import StockPortfolioEnv
from models import DRLAgent
from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts

import itertools


def train_stock_trading(dataset, K, strategy):
    """
    train an agent
    """

    print("==============Start Fetching Data===========")
    df, indexing_data = DataLoader(
        data_name=dataset,
    ).fetch_data_SP500()

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=False,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    )

    df = fe.preprocess_data(df)
    # add covariance matrix as states
    df=df.sort_values(['Week_ID','tic'],ignore_index=True)
    df.index = df.Week_ID.factorize()[0]
    return_list, price_list = [], []
    # look back is 120 for half year and 250 for whole year
    lookback=120
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      index_lookback = indexing_data.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close')
      price_list.append(price_lookback.values)    # price_list shape (lookback, stock_num)
      # return_lookback = df.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().fillna(0)
      # return_list.append(return_lookback.values)

      # index looaback used for optimization of QP
      # return_lookback = index_lookback.pivot_table(index = 'Week_ID',columns = 'tic', values = 'Close').pct_change().dropna()
      # index_lookback_list.append(index_lookback['Indexing'].values)
    # df_fuse = pd.DataFrame({'Week_ID':df.Week_ID.unique()[lookback:-1],'price_list':price_list,\
    #                     'return_list':return_list,'index_list':index_lookback_list})
    df_fuse = pd.DataFrame({'Week_ID':df.Week_ID.unique()[lookback:],'price_list':price_list})
    df = df.merge(df_fuse, on='Week_ID')
    df = df.sort_values(['Week_ID','tic']).reset_index(drop=True)

    # truth_indexing = indexing_data['Indexing'].pct_change().dropna()
    # indexing_data = indexing_data['Indexing'].pct_change().dropna().iloc[lookback:]

    # Training & Trading data split
    # train = data_split(df, config.START_DATE, config.START_TRADE_DATE) #train has 8 slice of data, test has 1 slice data, return_list contain 8 slices data, each slice contain 3 month
    train, trade, train_index, trade_index  = data_split_SP500(df, indexing_data, lookback)
    # train, trade = data_split(df, indexing_data, lookback)

    # calculate state action space
    stock_dimension = len(train[0].tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension-1,
        "reward_scaling": 1e-4,
        "lookback": lookback+1, #lookback=9, but the data length is 10
        }
    # clear the trained models file
    TRAINED_MODEL_PATH = "./" + config.TRAINED_MODEL_DIR  + "/" + dataset + "/"
    # TRAINED_MODEL = TRAINED_MODEL_PATH + 'best_model.pt'
    # shutil.rmtree(TRAINED_MODEL_PATH)
    for file_name in os.listdir(TRAINED_MODEL_PATH): os.remove(os.path.join(TRAINED_MODEL_PATH, file_name))
    # for tra, tes, train_ind, trade_ind in zip(train, trade, train_index, trade_index):

    for i, (tra, tes, train_ind, trade_ind) in enumerate(zip(train, trade, train_index, trade_index)):
        e_train_gym = StockPortfolioEnv(df=tra, indexing=train_ind, **env_kwargs)
        e_trade_gym = StockPortfolioEnv(df=tes, indexing=trade_ind, **env_kwargs)

        agent = DRLAgent(env=e_train_gym)

        print("==============Model Training Phase {}===========".format(i))
        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
        # =================== A2C=====================
        # model_a2c = agent.get_model("a2c")
        # if len(os.listdir(TRAINED_MODEL_PATH))>0: model_a2c.load(TRAINED_MODEL_PATH)
        # trained_a2c = agent.train_model(
        #     model=model_a2c, tb_log_name="a2c", total_timesteps=400, eval_env = e_trade_gym
        # )
        # trained_a2c.save(TRAINED_MODEL_PATH)


        # df_account_value, df_actions, df_error = DRLAgent.DRL_prediction(
        #     model=trained_a2c, environment = e_trade_gym
        # )

        #=================== DQN =====================
        model_dqn = agent.get_model("dqn", K, strategy)
        if len(os.listdir(TRAINED_MODEL_PATH))>0: model_dqn.load(TRAINED_MODEL_PATH)
        trained_dqn = agent.train_model(
            model=model_dqn, tb_log_name="dqn", total_timesteps=50, eval_env = e_trade_gym
        )
        trained_dqn.save(TRAINED_MODEL_PATH)


        print("==============Start Trading Phase {}===========".format(i))

        period_return_memory, days_return_memory = DRLAgent.DRL_prediction(
            model=trained_dqn, environment = e_trade_gym
        )


        try:
            df_daily_return = pd.concat([df_daily_return, days_return_memory.loc[1:,:]],ignore_index=True)
            df_period_return = pd.concat([df_period_return, period_return_memory.loc[1:,:]],ignore_index=True)
        except Exception as e:
            df_daily_return = days_return_memory.loc[1:,:]
            df_period_return = period_return_memory.loc[1:,:]
        # exit()

        # df_account_value.to_csv("./" + config.RESULTS_DIR + "/df_account_value_" + dataset + "{}.csv".format(i))
        # df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + dataset + "{}.csv".format(i))
        # df_daily_return.to_csv("./" + config.RESULTS_DIR + "/df_error_" + dataset + ".csv")

    print("==============Get Backtest Results===========")

    account_value_list = []
    period_value_list = df_period_return.daily_return.values
    print(df_daily_return['daily_return'].values[1:])
    # exit()
    for value in df_daily_return['daily_return'].values[1:]: account_value_list.append(value)
    # for period_value in df_period_return.values[1:]: period_value_list.extend(period_value)

    date_list = range(1,len(account_value_list)+1)
    period_list = range(1, len(period_value_list)+1)
    df_account_value = pd.DataFrame({"date":date_list,"daily_return":account_value_list})
    df_period_return_value = pd.DataFrame({"date":period_list,"daily_return":period_value_list})
    df_account_value = df_account_value.dropna()
    print(np.array(account_value_list),np.mean(account_value_list),np.std(account_value_list))
    # exit()
    df_account_value.to_csv("./" + config.RESULTS_DIR + "/df_daily_return_" + dataset + ".csv")

    from pyfolio import timeseries
    DRL_strat = convert_daily_return_to_pyfolio_ts(df_account_value)
    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
    print("==============DRL Strategy Stats===========")
    print(perf_stats_all)

    # DRL_strat2 = convert_daily_return_to_pyfolio_ts(df_period_return_value)
    # perf_stats_all2 = perf_func( returns=DRL_strat2,
    #                           factor_returns=DRL_strat2,
    #                             positions=None, transactions=None, turnover_denom="AGB")
