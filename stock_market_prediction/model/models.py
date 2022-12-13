# common library
import pandas as pd
import numpy as np
import time
import gym

from stable_baselines3 import PPO

# RL models from stable-baselines
# from stable_baselines import GAIL, SAC
# from stable_baselines import ACER
# from stable_baselines import PPO2
# from stable_baselines import A2C
# from stable_baselines import DDPG
# from stable_baselines import TD3

# from stable_baselines3.ddpg.policies import DDPGPolicy
# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines3.common.vec_env import DummyVecEnv
from preprocessing.preprocessors import *
from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

from ptr_ppo_common.ptr_ppo import PTR_PPO

import torch



def train_PTR_PPO(env_train, model_name, timesteps=50000):
    """PTR PPO model"""
    print(torch.cuda.is_available())
    start = time.time()
    model = PTR_PPO('MlpPolicy', env_train, ent_coef = 0.005, device="cuda")
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    """PPO model"""
    
    print(torch.cuda.is_available())
    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005, batch_size = 8, device="cuda")
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model



def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
    return sharpe


def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    last_state_ppo = []
    last_state_ptr_ppo = []
    ppo_sharpe_list = []
    ptr_ppo_sharpe_list = []

    model_use = []

    # based on the analysis of the in-sample data
    #turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #historical_turbulence = df[(df.datadate<unique_trade_date[i - rebalance_window - validation_window]) & (df.datadate>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]


        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
        print("turbulence_threshold: ", turbulence_threshold)

        ############## Environment Setup starts ##############
        ## training env
        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############





        print("======PTR PPO Training========")
        model_ptr_ppo = train_PTR_PPO(env_train, model_name="PTR_PPO_100k_dow_{}".format(i), timesteps=100000)
        print("======PTR PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ptr_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_ptr_ppo = get_validation_sharpe(i)
        print("PTR PPO Sharpe Ratio: ", sharpe_ptr_ppo)
        print("\n")



        # print("======PPO Training========")
        # model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        # print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
        #       unique_trade_date[i - rebalance_window])
        # DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
        # sharpe_ppo = get_validation_sharpe(i)
        # print("PPO Sharpe Ratio: ", sharpe_ppo)
        # print("\n")


        
        


        # ppo_sharpe_list.append(sharpe_ppo)

        ptr_ppo_sharpe_list.append(sharpe_ptr_ppo)

        
        


        ############## Trading starts ##############
        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)


        last_state_ptr_ppo = DRL_prediction(df=df, model=model_ptr_ppo, name="ptr_ppo",
                                             last_state=last_state_ptr_ppo, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        
        # last_state_ppo = DRL_prediction(df=df, model=model_ppo, name="ppo",
        #                                      last_state=last_state_ppo, iter_num=i,
        #                                      unique_trade_date=unique_trade_date,
        #                                      rebalance_window=rebalance_window,
        #                                      turbulence_threshold=turbulence_threshold,
        #                                      initial=initial)

        # print("============Trading Done============")
        ############## Trading ends ##############
        print("\n")
        print("\n")

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
