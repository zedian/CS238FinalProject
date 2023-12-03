import os
import csv
from typing import Callable, List, Optional
from utils.cache import Cache

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