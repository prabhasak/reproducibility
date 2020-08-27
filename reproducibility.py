"""Benchmark reinforcement learning (RL) algorithms from Stable Baselines 2.10. Verify experiment reproducibility
Author: Prabhasa Kalkur

- Note 1.0: choose {env, RL algo, training times, hyperparameters, etc} as cmd line arguments
- Note 1.1: changeable numbers in the program:
            callback model saving and evaluation = every 100 episodes for RL (line 252)
            number of episodes used for policy evaluation after training = 100 (line 288)
- Note 2: Things you can add on top: HP tuning, comparing consecutive runs of an experiment and retaining the better policy
"""

import sys
import os
import time
import argparse
import importlib
import warnings
from collections import OrderedDict
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

import gym
import yaml
import numpy as np
from pprint import pprint
import stable_baselines
from collections import deque

from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines import SAC, TRPO, DQN, PPO2, A2C, DDPG, ACER, ACKTR, HER, TD3
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.schedules import constfn
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy

from utils import get_wrapper_class, linear_schedule, make_env, StoreDict

from airsim_env.envs.airsim_env_0 import AirSim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
done_count, success_count, episode_reward, total_reward = 0, 0, 0, 0

algo_list = {'sac': SAC, 'trpo': TRPO, 'acer': ACER, 'dqn': DQN, 'ppo2': PPO2,
            'ddpg': DDPG, 'a2c': A2C, 'acktr': ACKTR, 'her': HER, 'td3': TD3}
env_list = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0', 'BipedalWalker-v3',
            'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0'] # mujoco envs need license
env_success = [-200, 475, 200, 200, 90, 300, 4800, 3000, 1000, 6000, 3.75, 360, 1000] # OpenAI Gym requirements (Hopper should be 3800)
episode_len = [200, 500, 400, 400, 999, 1600, 1000, 1000, 1000, 1000, 50, 1000, 100]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='CPU or GPU', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1', choices=env_list)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(algo_list.keys()))

    parser.add_argument('--exp-id', help='Experiment ID', default=0, type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=0, type=int)
    # parser.add_argument('-rl', '--train-RL', help='Train RL', action='store_true')
    parser.add_argument('-trl', '--timesteps-RL', help='Overwrite the number of timesteps for RL', default=-1, type=str)
    parser.add_argument('--test', help='Test trained policy', action='store_true')

    parser.add_argument('-tb', '--tensorboard', help='For Tensorboard logging', action='store_true') # tensorboard
    parser.add_argument('-check', '--check-callback', help='For saving models every save_freq steps', action='store_true') # checkpoint-callback
    parser.add_argument('-eval', '--eval-callback', help='For evaluating model every eval_freq steps', action='store_true') # eval-callback
    parser.add_argument('--n-envs', help='number of threads for multiprocessing', default=1, type=int, choices=list(range(13))) # multiprocessing
    parser.add_argument('-norm', '--normalize', help='Normalize env spaces', action='store_true') # with VecNormalize
    parser.add_argument('-m', '--monitor', help='Log episode length, rewards', action='store_true') # Monitor wrapper

    parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict, help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    
    args = parser.parse_args()
    return args

def evaluate(mode, quantity, env_id, env, algo, n_eval_episodes):
    env_index = env_list.index(env_id)
    if mode=='policy':
        print('\nTrained {} with {}. Some stats:\n'.format(env_id, algo))
        episode_return, episode_length = evaluate_policy(quantity, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        success_count = sum(i >= env_success[env_index] for i in episode_return)
        success_criteria = episode_return
    print('{}/{} successful episodes\n'.format(success_count, n_eval_episodes))
    print('Mean return: ', np.mean(episode_return))
    print('Std return: ', np.std(episode_return))
    print('Max return: ', max(episode_return))
    print('Min return: ', min(episode_return))
    print('Mean episode len: ', np.rint(np.mean(episode_length)))
    if np.mean(success_criteria) >= env_success[env_index]:
        print('Optimal {} found!\n'.format(mode))
    else:
        print('Suboptimal {}. Please try again...'.format(mode))
        sys.exit()

def check_success(env_index, env_success, success_count):
    if episode_reward>=env_success[env_index]:
        success_count+=1
        print('Test: Success!')
    else:
        print('Test: Fail!')
        pass
    return success_count

def moving_average(values, window):
    # Smooth values by doing a moving average
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):] # Truncate x
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(os.path.join(log_folder,'Learning_curve.png'))
    plt.show()

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def choose_device(device_name):
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

def main():

    args = get_args()
    choose_device(args.device)
    set_global_seeds(args.seed)

    env_id = args.env
    exp_id = args.exp_id
    algo = args.algo
    env_name = env_id[:-3]
    env_index = env_list.index(env_id)

    # Pass CustomEnv arguments: follow this for your CustomEnv if reward not known prior to training
    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs
    if (args.env_kwargs is not None) and (env_id in ['AirSim-v0']):
        if 'rew_land' in env_kwargs:
            if (int(env_kwargs['rew_land']) in [500, 1000, 10000]):
                env_success[-1] = int(env_kwargs['rew_land'])
            else:
                raise ValueError('Given env reward not acceptable. Please try again') 

    params = [exp_id, env_name.lower()]
    folder = [exp_id, env_name.lower(), args.algo.lower()]
    tensorboard_path, monitor_path, callback_path = None, None, None

    if args.tensorboard:
        tensorboard_path = "tensorboard/{}_{}".format(*params)
        make_dir(tensorboard_path)

    # if args.train_RL: # Begin training here (location of this condition also decides experiment performance)

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)

    # OPTIONAL: Print saved hyperparams
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    if args.verbose > 0:
        pprint(saved_hyperparams)

    if args.n_envs > 1:
        # if args.verbose:
        print("Overwriting n_envs with n={}".format(args.n_envs))
        n_envs = args.n_envs
    else:
        n_envs = hyperparams.get('n_envs', 1)

    # choose Monitor log path according to multiprocessing setting
    if args.monitor:
        if n_envs == 1:
            monitor_path = 'logs/single/{}_{}_{}'.format(*folder)
        else:
            if algo not in ['dqn', 'her', 'sac', 'td3']:
                monitor_path = 'logs/multi/{}_{}_{}'.format(*folder)
        make_dir(monitor_path)

    if int(float(args.timesteps_RL)) > 0:
        # if args.verbose:
        print("Overwriting n_timesteps with n={}".format(int(float(args.timesteps_RL))))
        n_timesteps = int(float(args.timesteps_RL))
    else:
        n_timesteps = int(hyperparams['n_timesteps'])

    # Convert to python object if needed
    if 'policy_kwargs' in hyperparams.keys() and isinstance(hyperparams['policy_kwargs'], str):
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps'] #To avoid error

    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']
    
    # if (algo=='ppo2' and ('learning_rate' in hyperparams.keys())):
    #     hyperparams['learning_rate'] = linear_schedule(hyperparams['learning_rate'])
        
    def create_env(n_envs, eval_env=False):
        if algo in ['a2c', 'acer', 'acktr', 'ppo2']:
            if n_envs > 1:
                env = SubprocVecEnv([make_env(env_id, i, args.seed, log_dir=monitor_path, wrapper_class=env_wrapper, env_kwargs=env_kwargs) for i in range(n_envs)])
            else:
                env = DummyVecEnv([make_env(env_id, 0, args.seed, log_dir=monitor_path, wrapper_class=env_wrapper, env_kwargs=env_kwargs)])
            env = DummyVecEnv([lambda: gym.make(env_id, **env_kwargs)])
            if env_wrapper is not None:
                env = env_wrapper(env)
        elif ((algo in ['dqn', 'her', 'sac', 'td3']) and n_envs > 1):
            raise ValueError("Error: {} does not support multiprocessing!".format(algo))
        elif ((algo in ['ddpg', 'ppo1', 'trpo', 'gail']) and n_envs > 1):
            raise ValueError("Error: {} uses MPI for multiprocessing!".format(algo))
        else:
            env = make_vec_env(env_id, n_envs=n_envs, seed=args.seed, monitor_dir=monitor_path, wrapper_class=env_wrapper, env_kwargs=env_kwargs)
        
        if args.normalize: # choose from multiple options
            # env = VecNormalize(env, clip_obs=np.inf)
            env = VecNormalize(env, norm_reward=False, clip_obs=np.inf)
            # env = VecNormalize(env, norm_reward=False, clip_obs=np.inf, **normalize_kwargs)
        return env
    
    # Zoo: env = SubprocVecEnv([make_env(env_id, i, seed, log_dir, wrapper_class=env_wrapper, env_kwargs=env_kwargs) for i in range(n_envs)])
    # Zoo: env = DummyVecEnv([make_env(env_id, 0, seed, log_dir, wrapper_class=env_wrapper, env_kwargs=env_kwargs)])
    env = create_env(n_envs)

    # if args.train_RL: # checking impact of the if-condition position on experiment reproducibility

    callback, callback_path = [], "callbacks/{}_{}_{}".format(*folder)
    save_freq, eval_freq = 100*episode_len[env_index], 100*episode_len[env_index]
    save_freq, eval_freq = max(save_freq // n_envs, 1), max(eval_freq // n_envs, 1)
    make_dir(callback_path)
    if args.check_callback:
        callback.append(CheckpointCallback(save_freq=save_freq, save_path=callback_path, name_prefix='rl_model', verbose=1))
    if args.eval_callback:
        callback.append(EvalCallback(create_env(1, eval_env=True), best_model_save_path=callback_path, log_path=callback_path, eval_freq=eval_freq, verbose=1))

    model = (algo_list[args.algo])(env=env, seed=args.seed, tensorboard_log=tensorboard_path, n_cpu_tf_sess=1, verbose=args.verbose, **hyperparams)
    print('\nTraining {} on {} now... \n'.format(algo, env_id))

    start_time = time.time()
    model.learn(total_timesteps=n_timesteps, callback=callback)
    total_time = time.time() - start_time

    if args.normalize:
        env.save(os.path.join(callback_path, "vec_normalize.pkl"))

    if n_envs > 1 or (algo in ['ddpg', 'trpo', 'gail']):
        print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time, n_timesteps / total_time))
    else:
        print("Took {:.2f}s for single process version - {:.2f} FPS".format(total_time, n_timesteps / total_time))

    env = DummyVecEnv([make_env(env_id, 0, args.seed, env_kwargs=env_kwargs)])
    
    if args.normalize:
        env = VecNormalize.load(os.path.join(callback_path, "vec_normalize.pkl"), env)
        env.training = False
        env.norm_reward = False
        env.seed(args.seed)

    # Evaluate RL model - choose either best model or last available model
    model = (algo_list[algo]).load(os.path.join(callback_path,'best_model'))
    # model = (algo_list[algo]).load("models/{}_{}_{}".format(*folder))
    model.set_env(env)    
    evaluate('policy', model, env_id, env, algo, 100)

    if args.monitor:
        results_plotter.plot_results([monitor_path], n_timesteps, results_plotter.X_TIMESTEPS, "{} {}".format(algo, env_id))
        plot_results(monitor_path)
    
    if args.test:
        print('\nTesting policy...\n')
        obs = env.reset()
        for _ in range(n_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, rewards, dones, info = env.step(action)
            episode_reward+=rewards
            env.render()
            if dones:
                done_count+=1
                success_count = check_success(env_index, env_success, success_count)
                total_reward+=episode_reward
                episode_reward = 0
                env.reset()
        print('\n{}/{} successful episodes'.format(success_count, done_count))
        average_reward = total_reward/done_count
        print('\nAverage reward: {}'.format(average_reward))
        env.close()

if __name__=='__main__':
    main()