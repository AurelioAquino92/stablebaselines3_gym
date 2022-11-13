import gym
import time
from stable_baselines3 import PPO

models_dir = "modelsCar/PPO"

env = gym.make('CarRacing-v0')  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/600000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs.copy())
		obs, rewards, done, info = env.step(action)
		env.render()
		time.sleep(0.001)
		print(obs)