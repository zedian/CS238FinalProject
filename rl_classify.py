import pandas as pd
from stable_baselines3 import PPO
import gym
from gym import spaces
import openai
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score


test_predictions_df = pd.read_csv('test_predictions_probabilities.csv')

test_embeddings_df = pd.read_csv('test_embeddings.csv')

class CustomEnv(gym.Env):
    def __init__(self, embeddings_shape):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(2) # 0 or 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=embeddings_shape, dtype=np.float32) 

    def reset(self):
        return np.zeros(self.observation_space.shape)

    def step(self, action):
        return np.zeros(self.observation_space.shape), 0.0, False, {}

    def render(self, mode='human'):
        pass

def load_policy(model_path, embeddings_shape):
    env = CustomEnv(embeddings_shape)
    policy = PPO.load(model_path, env=env)
    return policy

def adjust_probabilities(row, policy):
    embedding = test_embeddings_df.loc[test_embeddings_df['original_index'] == row['original_index']].drop(columns=['original_index']).to_numpy()[0]
    action, _states = policy.predict(embedding)
    adjusted_probabilities = [row['prob_0'], row['prob_1']]
    if action == 0:
        adjusted_probabilities[0] += 0.15  # Increase probability of 0
        adjusted_probabilities[1] -= 0.15 # Decrease probability of 1
    elif action == 1:
        adjusted_probabilities[0] -= 0.15  # Decrease probability of 0
        adjusted_probabilities[1] += 0.15  # Increase probability of 1
    adjusted_probabilities = [p / sum(adjusted_probabilities) for p in adjusted_probabilities]
    return 1 if adjusted_probabilities[1] > adjusted_probabilities[0] else 0

def evaluate_rl_adjusted(df, policy):
    df['rl_adjusted_predicted'] = df.apply(lambda row: adjust_probabilities(row, policy), axis=1)
    precision = precision_score(df['label'], df['rl_adjusted_predicted'])
    recall = recall_score(df['label'], df['rl_adjusted_predicted'])
    accuracy = accuracy_score(df['label'], df['rl_adjusted_predicted'])
    return precision, recall, accuracy

policy = load_policy("ppo_model2.zip", (1537,))

precision, recall, accuracy = evaluate_rl_adjusted(test_predictions_df, policy)
print(f"RL-adjusted classification results: Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
