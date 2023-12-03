import os
import csv
import re
from typing import Callable, List, Optional
from utils.cache import Cache
from transformers import GPT2Tokenizer

class CommonsenseData:
    def __init__(self, path: str,
                indexes: Optional[List[int]]=None,
                file_name: Optional[str]="example.csv",
                reward_f: Optional[Callable[[str], float]]=None,
                reward_cache: Optional[Cache]=None, 
                reward_shift: Optional[float]=0.0, 
                reward_scale: Optional[float]=1.0):
        with open(os.path.join(path, file_name), 'r') as f:
            items = [row for row in csv.reader(f)]
        if indexes is not None:
            items = [items[idx] for idx in indexes]
        # self.info = (path, len(indexes))
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()
        self.contexts = [item[0] for item in items]
        self.acts = [item[1] for item in items]
        self.demonstrations = [item[2] for item in items]
        self.rewards = [item[3] for item in items]
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale

        # TODO remove this is not needed.
        self.reward_f = reward_f

    def __getitem__(self, idx):
        ctx = self.contexts[idx]
        act = self.acts[idx]
        demonstration = self.demonstrations[idx]
        reward = float(self.rewards[idx])
        if demonstration not in self.reward_cache:
            self.reward_cache[demonstration] = reward
        return (ctx, act, demonstration), self.reward_cache[demonstration] * self.reward_scale + self.reward_shift
    
    
    def __len__(self):
        return len(self.contexts)
    
class CommonsenseSequenceData:
    def __init__(self, path: str,
                indexes: Optional[List[int]]=None,
                file_name: Optional[str]="example.csv",
                reward_f: Optional[Callable[[str], float]]=None,
                reward_cache: Optional[Cache]=None, 
                reward_shift: Optional[float]=0.0, 
                reward_scale: Optional[float]=1.0):
        with open(os.path.join(path, file_name), 'r') as f:
            items = [row for row in csv.reader(f)]
        if indexes is not None:
            items = [items[idx] for idx in indexes]
        # self.info = (path, len(indexes))
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()
        self.contexts = [item[0] for item in items]
        self.demonstrations = [item[2] for item in items]
        self.state_action_next_state = []
        
        for dem in self.demonstrations:
            for l in self.preprocess_states(dem):
                self.state_action_next_state.append(l)

        self.acts = [item[1] for item in items]
        # Can ignore this use deterministic reward instead
        self.rewards = [item[3] for item in items] 
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.reward_f = reward_f

    def preprocess_states(self, str):
        result = []

        substrings = str.split("AI: ")
        prefix = substrings[0] + "AI: "
        for i in range(1, len(substrings)):
            parts = substrings[i].split("H: ")
            words = re.findall(r'\b\w+\b|[.,;\'!?]', parts[0])

            for word in words:
                result.append((prefix, word, prefix + word))
                prefix += word + " "
            
            if len(parts) > 1:
                prefix += parts[1]

        return result

    def __getitem__(self, idx):
        # state = self.states[idx]
        # action = self.action[idx]

        state, action, next_state = self.state_action_next_state[idx]
        # next_state = state + action
        # reward = float(self.rewards[idx])

        # next state is just (state, action)
        if next_state not in self.reward_cache:
            self.reward_cache[next_state] = self.reward_f(state) if self.reward_f is not None else 0.0
        return (state, action, next_state), self.reward_cache[next_state] * self.reward_scale + self.reward_shift
    
    
    def __len__(self):
        return len(self.contexts)