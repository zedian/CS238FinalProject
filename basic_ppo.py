import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

original_df = pd.read_csv('output_dialogues_all.csv')

embeddings_df = pd.read_csv('embeddings.csv')
labels_df = pd.read_csv('output_dialogues_all.csv')['label']

X_train, X_test, y_train, y_test = train_test_split(embeddings_df, labels_df, test_size=0.2)

class CustomEnv(gym.Env):
    def __init__(self, embeddings, labels, reward_function):
        super(CustomEnv, self).__init__()
        self.embeddings = embeddings.to_numpy()
        self.labels = labels.to_numpy()
        self.reward_function = reward_function
        self.action_space = spaces.Discrete(2) #0 or 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(embeddings.shape[1],), dtype=np.float32)
        self.current_index = 0
        self.step_count = 0
        self.max_steps = len(self.embeddings) * 4 # Increase the number of steps per episode

    def reset(self):
        self.current_index = np.random.randint(0, len(self.embeddings))
        self.step_count = 0
        return self.embeddings[self.current_index]

    def step(self, action):
        self.step_count += 1
        label = self.labels[self.current_index]
        reward = self.reward_function(action, label)
        self.current_index = np.random.randint(0, len(self.embeddings))
        done = self.step_count >= self.max_steps
        return self.embeddings[self.current_index], reward, done, {}

    def render(self, mode='human'):
        pass

def reward_function(action, label):
    return 1.0 if action == label else -1.0

env = CustomEnv(X_train, y_train, reward_function)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)

total_timesteps = 20000 

with tqdm(total=total_timesteps, desc="Training progress") as pbar:
    steps_completed = 0
    while steps_completed < total_timesteps:
        result = model.learn(total_timesteps=min(model.n_steps, total_timesteps - steps_completed), reset_num_timesteps=False)
        steps_completed += result.n_steps
        pbar.update(result.n_steps)

num_episodes = 100
env_test = CustomEnv(X_test, y_test, reward_function)
env_test = DummyVecEnv([lambda: env_test])
total_rewards = 0
for _ in range(num_episodes):
    obs = env_test.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env_test.step(action)
        env_test.render()
        total_rewards += rewards[0]
        done = dones[0]

average_reward = total_rewards / num_episodes
print(f"avg reward over {num_episodes} episodes: {average_reward}")

model.save("ppo_model2.zip")
