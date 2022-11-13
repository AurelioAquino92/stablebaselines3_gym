import gym
import time
from stable_baselines3 import PPO

models_dir = "modelsAcrobot/PPO"

env = gym.make('Acrobot-v1')
env.reset()

model_path = f"{models_dir}/5620000.zip"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs.copy())
		obs, rewards, done, info = env.step(action)
		env.render()
		time.sleep(0.03)
		# print(obs)