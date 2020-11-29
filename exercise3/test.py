from env import PendulumEnv

env = PendulumEnv()
env.reset()
for t in range(500):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
env.close()

    