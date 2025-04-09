import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os


env = gym.make("Hopper-v5") #render_mode="human" add this to show the hopper
env = Monitor(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hopper_tensorboard/")
model.learn(total_timesteps=200_000)

os.makedirs("models", exist_ok=True)
model.save("models/ppo_hopper2")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()

#Use this in another terminal to see the graphs live
#tensorboard --logdir=ppo_hopper_tensorboard/