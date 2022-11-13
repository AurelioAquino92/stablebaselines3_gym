import gym
from stable_baselines3 import PPO

models_dir = "C:\\Users\\mnsaaqui\\Desktop\\python\\IA\\stablebaselines3_gym\\modelsCartPole\\PPO\\180000.zip"

env = gym.make('CartPole-v1')
env.reset()

model_path = f"{models_dir}/180000"
model = PPO.load(models_dir, env=env)

episodes = 5

for ep in range(episodes):
	obs = env.reset()
	done = False
	total_rew = 0
	while not done:
		action, _states = model.predict(obs.copy())
		obs, rewards, done, info = env.step(action)
		env.render()
		total_rew += rewards
	print(total_rew)