import random

from typing import List
from collections import defaultdict
from commonsense_data import CommonsenseData

def deterministic_reward(reddit_path: str, indexes: List[int]):
    data = CommonsenseData("Implicit-Language-Q-Learning/data/commonsense/")
    data_index = defaultdict(list)

    # TODO: there is likely only 1 value per key
    for idx, ((_, _, c), r) in enumerate(data):
        data_index[c].append(r)
    def _deterministic_reward(text: str):
        if text in data_index:
            return 2 * float(random.choice(data_index[text]) > 0) - 1
        raise NotImplementedError
    return _deterministic_reward