from toxicity.reddit_comments_base import RedditData
from utils.misc import convert_path
from toxicity.toxicity_env import ToxicityEnvironment
from toxicity.reward_fs import toxicity_reward, score_human_reward

# reward_f = score_human_reward(
#    reddit_path=convert_path('Implicit-Language-Q-Learning/data/reddit_comments/'),
#    indexes=None
# )

# reward_f

from commonsense_data import CommonsenseData

data = CommonsenseData("Implicit-Language-Q-Learning/data/commonsense/")
data[0]