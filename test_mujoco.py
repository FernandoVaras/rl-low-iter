import gymnasium as gym

env = gym.make("Walker2d-v4")
obs, info = env.reset()
print("✅ MuJoCo OK")
print(f"Obs shape: {obs.shape}")

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward:.4f}")

env.close()
print("✅ Todo funciona!")


