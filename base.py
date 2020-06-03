import pdb
import sys
import os
import argparse
import pkg_resources
import time
import uuid
import difflib
import importlib
import warnings
import random
from shutil import copy
from pprint import pprint
from collections import OrderedDict
import tensorflow as tf

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

import gym
import yaml
import numpy as np
import stable_baselines
from collections import deque
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.common.schedules import constfn
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList

from stable_baselines_zoo.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model
from stable_baselines_zoo.utils import make_env, linear_schedule, get_wrapper_class
from stable_baselines_zoo.utils.hyperparams_opt import hyperparam_optimization
from stable_baselines_zoo.utils.utils import StoreDict
from stable_baselines_zoo.utils.callbacks import SaveVecNormalizeCallback

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import GAIL, SAC, TRPO, DQN, PPO2, A2C, DDPG, ACER, ACKTR, HER, TD3
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from airsim_env.envs.airsim_env_0 import AirSim

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# path = '/home/prabhasa1/Desktop/GAIL-TF-SB' # Lab system - Ubuntu
path = 'C:/IL_UAV/GAIL-TF-SB' # Laptop - new
# path = 'G:/UAV/GAIL-TF-SB' # Laptop - old
# path = 'C:/Prabhasa/GAIL-TF-SB' # Lab system - Windows

done_count, success_count, episode_reward, total_reward = 0, 0, 0, 0

path_logs = path+'/data/openai/logs/'
path_logs_test = path+'/data/openai/logs/test'
path_model_RL_temp = path+'/data/openai/models/RL_temp/'
path_model_RL = path+'/data/openai/models/RL/'
path_model_RL_final = path+'/data/openai/models/RL/done/'
path_expert_temp = path+'/data/openai/expert/expert_temp/'
path_expert = path+'/data/openai/expert/'
path_expert_final = path+'/data/openai/expert/done/'
path_tensorboard = path+'/data/openai/tensorboard/'
path_callback_RL = path+'/data/openai/callback/RL_model/'

airsim_success_reward = 500

algo_list = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'gail': GAIL
}
env_list = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v2', 'BipedalWalkerHardcore-v2',
            'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0']
env_success = [-200, 475, 200, 200, 300, 300, 9000, 3000, 1000, 6000, 3.75, 360, airsim_success_reward]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1', choices=env_list)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('--device', help='CPU or GPU', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('-g', '--generated-trajs', help='Done generating expert data', default=1, type=int, choices=[0, 1])
    parser.add_argument('-trl', '--train-RL-model', help='RL training done', default=0, type=int, choices=[0, 1])
    parser.add_argument('-til', '--train-IL-model', help='GAIL training done', default=0, type=int, choices=[0, 1])
    parser.add_argument('-s', '--testing', help='Testing learnt model', default=0, type=int, choices=[0, 1])
    parser.add_argument('-f', '--folder', help='Expert log folder', type=str, default=path_expert_temp)
    parser.add_argument('-l', '--log-folder', help='Log folder', type=str, default=path_logs)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training', default='', type=str)
    
    parser.add_argument('-RL', '--timesteps-RL', help='Overwrite the number of timesteps for RL', default=-1, type=str)
    parser.add_argument('-GAIL', '--timesteps-GAIL', help='Overwrite the number of timesteps for GAIL', default=-1, type=str)
    parser.add_argument('-test', '--timesteps-test', help='number of timesteps for testing', default="1e4", type=str)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)

    parser.add_argument('--eval-freq-RL', help='Evaluate the RL agent every n steps (if negative, no evaluation)', default=10000, type=int)
    parser.add_argument('--eval-freq-IL', help='Evaluate the IL agent every n steps (if negative, no evaluation)', default=100000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation', default=5, type=int)
    parser.add_argument('--save-freq-RL', help='Save the RL model every n steps (if negative, no checkpoint)', default=-1, type=int)
    parser.add_argument('--save-freq-IL', help='Save the model every n steps (if negative, no checkpoint)', default=-1, type=int)
    parser.add_argument('-r', '--rew-threshold', help='To stop training based on reward and not timesteps', default=0, type=int)

    parser.add_argument('-eval', '--n-eval-episodes', help='number of episodes for evaluating the GAIL model', default=100, type=int)
    parser.add_argument('-gen', '--expert-traj-gen', help='number of expert trajectories to be generated by RL', default=100, type=int)
    parser.add_argument('-use', '--expert-traj-use', help='number of expert trajectories to be used by GAIL', default=10, type=int)

    parser.add_argument('--n-envs', help='number of environments', default=1, type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=0, type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=0, type=int)
    parser.add_argument('--no-render', action='store_true', default=False, help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False, help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--load-best', action='store_true', default=False, help='Load best model instead of last model if available')
    parser.add_argument('--norm-reward', action='store_true', default=False, help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('-uuid', '--uuid', action='store_true', default=False, help='Ensure that the run has a unique ID')

    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False, help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str, default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str, default='median', choices=['halving', 'median', 'none'])
    
    parser.add_argument('-params-RL', '--hyperparams', type=str, nargs='+', action=StoreDict, help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('-params-GAIL', '--hyperparams-GAIL', type=str, nargs='+', action=StoreDict, help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('--reward-log', help='Where to log reward', default=path_logs, type=str)
    parser.add_argument('--env-kwargs-RL', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    parser.add_argument('--env-kwargs-IL', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    args = parser.parse_args()

    return args

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_global_seeds(seed)
    tf.set_random_seed(seed) #tf.random.set_seed(seed) for TF 2.0
    np.random.seed(seed)
    random.seed(seed)

def init_args(args, algo, env_id):
    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

if __name__=='__main__':

    args = get_args()

    env_id = args.env
    exp_id = args.exp_id
    algo = args.algo
    n_envs = args.n_envs
    env_name = env_id[:-3]
    env_index = env_list.index(env_id)

    init_args(args, algo, env_id)
    set_seed(args.seed)
    # choose_device(args.device)

    params = [env_name, exp_id]
    params_save_RL = [env_name, algo, exp_id]
    tensorboard_RL = {'tensorboard_log': path_tensorboard+"{}_v{}".format(*params)}

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    if int(float(args.timesteps_RL)) > 0:
        # if args.verbose:
        print("Overwriting n_timesteps with n={}".format(int(float(args.timesteps_RL))))
        n_timesteps = int(float(args.timesteps_RL))
    else:
        n_timesteps = int(hyperparams['n_timesteps'])
    
    del hyperparams['n_timesteps'] #To avoid error

    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']
        
    env_kwargs = {} if args.env_kwargs_RL is None else args.env_kwargs_RL

    def create_env(n_envs, eval_env=False):

        log_dir = None

        if algo in ['dqn', 'ddpg']:
            env = gym.make(env_id, **env_kwargs)
            env.seed(args.seed)
            if env_wrapper is not None:
                env = env_wrapper(env)
        else:
            if n_envs == 1:
                env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper, log_dir=log_dir, env_kwargs=env_kwargs)])
            else:
                env = DummyVecEnv([make_env(env_id, i, args.seed, log_dir=path_logs,
                                            wrapper_class=env_wrapper, env_kwargs=env_kwargs) for i in range(n_envs)])

        return env
    
    env = create_env(n_envs)

    callbacks = []
    if args.save_freq_RL > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq_RL // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                                            save_path=path_callback_RL+'{}_{}_v{}'.format(*params_save_RL), name_prefix='rl_model', verbose=1))

    # Create test env if needed, do not normalize reward
    # eval_env = None
    if args.eval_freq_RL > 0:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq_RL // n_envs, 1)

        if args.rew_threshold:
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=env_success[env_index], verbose=1)
            callbacks.append(EvalCallback(create_env(1, eval_env=True), best_model_save_path=path_callback_RL+"{}_{}_v{}".format(*params_save_RL),
                                        log_path=path_callback_RL+"{}_{}_v{}".format(*params_save_RL), eval_freq=args.eval_freq, callback_on_new_best=callback_on_best, verbose=1))
        else:
            callbacks.append(EvalCallback(create_env(1, eval_env=True), best_model_save_path=path_callback_RL+"{}_{}_v{}".format(*params_save_RL),
                                                log_path=path_callback_RL+"{}_{}_v{}".format(*params_save_RL), eval_freq=args.eval_freq, verbose=1))

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}
    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    # env = gym.make(env_id, **env_kwargs)
    # env.seed(args.seed)

    model = (algo_list[args.algo])(env=env, seed=args.seed, n_cpu_tf_sess=1, **tensorboard_RL, verbose=args.verbose, **hyperparams)
    print('\nTraining {} on {} now... \n'.format(algo, env_id))
    model.learn(total_timesteps=n_timesteps, **kwargs)
    model_new = model.load(path_callback_RL+"{}_{}_v{}".format(*params_save_RL)+'/best_model')

    eval_episode_reward, eval_episode_len = evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes, return_episode_rewards=True)
    print('\nMean return: ', np.mean(eval_episode_reward))
    print('Std return: ', np.std(eval_episode_reward))
    print('Max return: ', max(eval_episode_reward))
    print('Min return: ', min(eval_episode_reward))
    print('Mean episode len: ', np.rint(np.mean(eval_episode_len)))
    eval_success_count = sum(i >= env_success[env_index] for i in eval_episode_reward)
    print('{}/{} successful episodes'.format(eval_success_count, args.n_eval_episodes))
    if np.mean(eval_episode_reward)>=env_success[env_index]:
        print('\nTrained {} model successful on {} as per OpenAI Gym requirements!'.format(algo, env_id))
