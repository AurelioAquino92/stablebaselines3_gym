import gym
from stable_baselines3 import PPO
import os

models_dir = "modelsAcrobot/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('Acrobot-v1') 
env.reset()

iters = 12
model_path = f"{models_dir}/{iters}0000.zip"
model = PPO.load(model_path, env=env, tensorboard_log=logdir)
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

