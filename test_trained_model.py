import gymnasium as gym
from stable_baselines3 import PPO

# Load environment with rendering
env = gym.make("Hopper-v5", render_mode="human")

# Load the saved model
model = PPO.load("models/ppo_hopper2")

obs, info = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()