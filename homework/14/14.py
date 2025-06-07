import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

episode_length = 0
episode_count = 0
max_steps = 1000

while True:
    env.render()
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation

    # 固定策略
    if pole_angle + 0.5 * pole_ang_vel > 0:
        action = 1  # 推右
    else:
        action = 0  # 推左

    observation, reward, terminated, truncated, info = env.step(action)
    episode_length += 1

    if terminated or truncated:
        print(f"🎯 Episode {episode_count+1} 結束，撐了 {episode_length} 步")
        episode_length = 0
        episode_count += 1
        observation, info = env.reset()
    
    if episode_count >= 3:
        break

env.close()
