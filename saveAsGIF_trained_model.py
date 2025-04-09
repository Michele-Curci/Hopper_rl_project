import gymnasium as gym
from stable_baselines3 import PPO
import imageio
import os

# Load the trained model
model = PPO.load("models/ppo_hopper2")

# Create the environment with rgb_array rendering
env = gym.make("Hopper-v5", render_mode="rgb_array")

# Set up recording
frames = []
obs, info = env.reset()
done = False

while not done:
    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Capture frame
    frame = env.render()
    frames.append(frame)

env.close()

# Save as GIF
gif_path = "hopper_run.gif"
imageio.mimsave(gif_path, frames, fps=30)
print(f"Saved GIF at: {gif_path}")