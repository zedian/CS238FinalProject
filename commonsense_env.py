from typing import Any, Callable, Dict, Optional, Tuple, List
from data.language_environment import Language_Environment, Language_Observation, Policy
from data.rl_data import List_RL_Dataset, Iterable_RL_Dataset, RL_Dataset
from commonsense_data import CommonsenseData
import random

class CommonsenseObservation(Language_Observation):
    def __init__(self, act: Optional[str], demonstration: Optional[str], reward: Optional[float]):
        assert (demonstration is None and reward is None) or (demonstration is not None and reward is not None)
        self.act = act
        self.demonstration = demonstration
        self.reward = reward
    
    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        if self.demonstration is None:
            if self.act is not None:
                return [(self.act, None)], False
            return [], False
        if self.act is None:
            return [(self.demonstration, self.reward)], True
        return [(self.act, None), (self.demonstration, self.reward)], True
    
    def __str__(self) -> str:
        if self.act is not None:
            return f'act: {self.act}\ndemonstration: {self.demonstration}'
        return self.demonstration

class CommmonsenseEnvironment(Language_Environment):
    def __init__(self, data: CommonsenseData, 
                 reward_f: Optional[Callable[[str], float]]=None, 
                 reward_shift: float=0.0, reward_scale: float=1.0, 
                 include_parent: bool=True):
        self.data = data
        self.reward_f = reward_f
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.include_parent = include_parent
        self.stepped = False
        self.parent = None
        self.reset()

    def step(self, action: str) -> Tuple[CommonsenseObservation, float, bool]:
        if self.stepped:
            raise Exception("Cannot step after final action")
        self.stepped = True
        reward = (self.reward_f(action) if self.reward_f is not None else 0.0) * self.reward_scale + self.reward_shift
        return CommonsenseObservation(self.parent, action, reward), reward, True

    def reset(self) -> CommonsenseObservation:
        self.stepped = False
        self.parent = None
        if self.include_parent:
            self.parent = random.choice(self.data)[0][0]
        return CommonsenseObservation(self.parent, None, None)

    def is_terminal(self) -> bool:
        return self.stepped