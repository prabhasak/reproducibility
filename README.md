A Reinforcement Learning benchmark: Experiment Reproducibility
==========================
**Objective:** Benchmark Reinforcement Learning algorithms from [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/) on OpenAI Gym problems. The code focuses on verifying reproducibility of results with a fixed random seed

**Idea**: Pick your favourite [env, algo] pair -> **train RL** -> evaluate learned policy

**Framework, langauge, OS:** Tensorflow 1.14, Python 3.7, Windows 10


Prerequisites
-------------
The implementation uses [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/guide/install.html). I have included the ``utils`` and ``hyperparams`` folder from [Baselines Zoo](https://github.com/araffin/rl-baselines-zoo)

```
# create virtual environment (optional)
conda create -n myenv python=3.7
conda activate myenv

git clone https://github.com/prabhasak/reproducibilty.git
cd reproducibility
git clone https://github.com/araffin/rl-baselines-zoo
pip install -r requirements.txt # recommended
pip install stable-baselines[mpi] # MPI needed for TRPO
```

**For CustomEnvs:** [Register](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) your CustomEnv on Gym ([examples](https://github.com/openai/gym/blob/master/gym/envs/__init__.py)), and add your custom env and/or algorithm details. You can use the "airsim_env" folder for reference


Usage
-------------
``python reproducibility.py --seed 42 --env Pendulum-v0 --algo sac -trl 1e5 -tb -check -eval -m -params learning_starts:1000``

**Verify reproducibility:** (i) 65/100 successful episodes on SAC policy evaluation, with (mean, std) = (-161.68, 86.39)

Features
-------------
1. **[Tensorboard](https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html)**: monitor performance during training (``tensorboard --logdir "/your/path"``)
2. **[Callbacks](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html)**:
  a. [Saving](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback) the model periodically (useful for [continual learning](https://stable-baselines.readthedocs.io/en/master/guide/examples.html#continual-learning) and to resume training)
  b. [Evaluating](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#evalcallback) the model periodically and saves the best model throughout training (you can choose to save and evaluate just the best model with ``-best``)
3. **[Multiprocessing](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv):** speed up training (observed 6x speedup for CartPole-v0 on my CPU with 12 threads). Note: TRPO uses MPI, so has multiprocessing enabled by default
4. [VecNormalize](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize): normalize env observation, action spaces (useful for MuJoCo environments)
5. [Monitor](https://stable-baselines.readthedocs.io/en/master/common/monitor.html): record internal state information during training (episode length, rewards). You can save a plot of the episode reward by modifying [``results_plotter.py``](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/results_plotter.py#L95)
6. Passing arguments to your CustomEnv, and loading hyperparameters from [Baselines Zoo](https://github.com/araffin/rl-baselines-zoo) (some are tuned)

Check out my [imitation learning repo](https://github.com/prabhasak/masters-thesis/blob/master/imitation_learning_basic.py)!