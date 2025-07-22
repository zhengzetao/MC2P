import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
matplotlib.use("Agg")
import datetime
import time

import config
from preprocessor.dataloader import DataLoader
from preprocessor.preprocessors import FeatureEngineer, data_split, series_decomposition
from env.env_portfolio import StockPortfolioEnv
from env.env_supplier import SupplierSelectionEnv
from models import DRLAgent
from plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts
import pdb
import itertools


def train_supplier_selection(dataset, K, strategy):
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df_train, demand_train = DataLoader(
        data_name = dataset.replace("test", "train"),
    ).fetch_data_SSOA()

    df_test, demand_test = DataLoader(
        data_name=dataset,
    ).fetch_data_SSOA()
    
    quan_price = ['MaxQuantity1', 'Price1']
    target_columns = ['GreenW', 'GeneralW', 'MinQuantity1', 'Price0', 'MaxQuantity0', 'CCCap', 'SP', 'CF', 'MaxQuantity1', 'Price1']
    scaler = StandardScaler()
    new_columns_order = [col for col in df_train.columns if col not in quan_price] + quan_price
    order_wo_supplier = original_order = [col for col in new_columns_order if col != 'supplier']
    df_train = df_train[new_columns_order]
    df_test = df_test[new_columns_order]
    # df_train = pd.DataFrame(scaler.fit_transform(df_train),columns=['GreenW', 'GeneralW', 'MinQuantity1', 'Price0', 'MaxQuantity0', 'CCCap', 'CF', 'MaxQuantity1', 'Price1'])
    max_supplier = max(max(df_train['supplier'].unique()),max(df_test['supplier'].unique()))
    # train data processing
    bool_terminal, supplier_list, feature_list, demand_train_list, price_list, quantity_list = [], [], [], [], [], []
    for i in df_train['data_id'].unique():
        df_train_data = df_train[df_train['data_id'] == i]
        demand_data = demand_train[demand_train['data_id'] == i]
        period_id = df_train_data['period_id'].unique()
        for j in df_train_data['period_id'].unique():
            df_train_period = df_train_data[df_train_data['period_id'] == j]
            price_list.append(df_train_period['Price0'].values)
            quantity_list.append(df_train_period['MaxQuantity0'].values)
            df_train_period = df_train_period.copy()
            df_train_period.loc[:,target_columns] = scaler.fit_transform(df_train_period.loc[:,target_columns])
            current_supplier_num = max(df_train_period['supplier'].unique())
            supplier_list.append(current_supplier_num)
            # padding the num of supplies
            if current_supplier_num < max_supplier:
                df_train_period = pd.concat([df_train_period, pd.DataFrame(0.0, index = range(max_supplier-current_supplier_num), columns=df_train_period.columns)],ignore_index=True)
                df_train_period.loc[current_supplier_num:max_supplier,['supplier']] = range(current_supplier_num+1,max_supplier+1)
            df_train_table = df_train_period.pivot_table(index = ['supplier'])[order_wo_supplier]
            df_train_table = df_train_table.drop(columns=['S','H','data_id','period_id','MinQuantity0','GreenW', 'GeneralW', 'MinQuantity1', 'Price0', 'Price1', 'MaxQuantity1', 'MaxQuantity0', 'CCCap'],errors='ignore')
            demand_period = demand_data[demand_data['period_id'] == j]
            feature_list.append(df_train_table.values)
            demand_train_list.append(demand_period['Demand'].values)
            bool_terminal.append(j == period_id[-1])
        # id_list.extend(period_id)
    train_dataframe = pd.DataFrame({'terminal':bool_terminal, 'supplier_num': supplier_list, 'feature':feature_list, 'price': price_list, 'quantity': quantity_list})
    # test data processing
    bool_terminal, supplier_list, feature_list, demand_test_list, price_list, quantity_list = [], [], [], [], [], []
    for i in df_test['data_id'].unique():
        df_test_data = df_test[df_test['data_id'] == i]
        demand_data = demand_test[demand_test['data_id'] == i]
        period_id = df_test_data['period_id'].unique()
        for j in df_test_data['period_id'].unique():
            df_test_period = df_test_data[df_test_data['period_id'] == j]
            price_list.append(df_test_period['Price0'].values)
            quantity_list.append(df_test_period['MaxQuantity0'].values)
            df_test_period = df_test_period.copy()
            df_test_period.loc[:,target_columns] = scaler.fit_transform(df_test_period.loc[:,target_columns])
            current_supplier_num = max(df_test_period['supplier'].unique())
            supplier_list.append(current_supplier_num)
            #padding the num of suppliers
            if current_supplier_num < max_supplier:
                df_test_period = pd.concat([df_test_period, pd.DataFrame(0, index = range(max_supplier-current_supplier_num), columns=df_test_period.columns)],ignore_index=True)
                df_test_period.loc[current_supplier_num:max_supplier,['supplier']] = range(current_supplier_num+1,max_supplier+1)
            df_test_table = df_test_period.pivot_table(index = ['supplier'])[order_wo_supplier]
            df_test_table = df_test_table.drop(columns=['S','H','data_id','period_id','MinQuantity0','GreenW', 'GeneralW', 'MinQuantity1', 'Price0','Price1','MaxQuantity1', 'MaxQuantity0', 'CCCap'],errors='ignore')
            demand_period = demand_data[demand_data['period_id'] == j]
            feature_list.append(df_test_table.values)
            demand_test_list.append(demand_period['Demand'].values)
            bool_terminal.append(j == period_id[-1])
        # id_list.extend(period_id)
    test_dataframe = pd.DataFrame({'terminal':bool_terminal, 'supplier_num': supplier_list, 'feature':feature_list, 'price': price_list, 'quantity': quantity_list})

    # calculate state action space
    # stock_dimension = len(train.tic.unique())
    supplier_num = max_supplier
    assert train_dataframe['feature'][0].shape[1] == test_dataframe['feature'][0].shape[1]
    state_space = train_dataframe['feature'][0].shape[1]
    print(f"Supplier Dimension: {supplier_num}, State Space: {state_space}")

    env_kwargs = {
        "initial_shortage": 0, 
        "state_space": state_space, 
        "supplier_num": supplier_num, 
        # "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        # "action_space": config.SELECTED_STOCK_NUM,
        "action_space": supplier_num,
        "reward_scaling": 1e-4,
        # "lookback": lookback+1, #lookback=9, but the data length is 10
        }
    TRAINED_MODEL_PATH = "./" + config.TRAINED_MODEL_DIR  + "/" + dataset + "/"

    e_train_gym = SupplierSelectionEnv(df=train_dataframe, demanding=demand_train_list, **env_kwargs)
    e_test_gym = SupplierSelectionEnv(df=test_dataframe, demanding=demand_test_list, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=e_train_gym)

    print("==============Model Training===========")
    ################ A2C #########################
    # model_a2c = agent.get_model("a2c",K)
    # trained_a2c = agent.train_model(
    #     model=model_a2c, tb_log_name="a2c", total_timesteps=2000, eval_env = e_trade_gym
    # )

    ################ DQN #########################
    model_dqn = agent.get_model("dqn", K, strategy)
    trained_dqn = agent.train_model(
        model=model_dqn, tb_log_name="dqn", total_timesteps=250000, eval_env = e_test_gym, model_save_path=TRAINED_MODEL_PATH
    )
    # trained_dqn.save(TRAINED_MODEL_PATH)

    print("============== Start Testing===========")
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    df_account_value, df_actions, df_error= DRLAgent.DRL_prediction(
        model=trained_dqn, environment = e_test_gym
    )
    total_time = time.time() - start_time
    print(f"Average Decision Time per Period: {total_time/df_actions.shape[0]:.6f} seconds")
    # df_account_value.to_csv(
    #     "./" + config.RESULTS_DIR + "/df_account_value_" + dataset + ".csv"
    # )
    # df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + dataset + ".csv")
    # df_error.to_csv("./" + config.RESULTS_DIR + "/df_error_" + dataset + ".csv")
