import numpy as np
import itertools
import random
import torch
import io


class ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,
                 n_features,
                 noise_std=1.0,
                 seed=None,
                 articles=None,
                 log_file=None,
                 ):
        # if not None, freeze seed for reproducibility
        self._seed(seed)
        self.article_keys = list(articles.keys())
        # number of rounds
        self.T = T
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features
        self.articles = articles
        self.log_file = log_file
        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std

        # generate random features
        self.reset()

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        with io.open(self.log_file, 'rb', buffering=1024 * 1024 * 512) as input_generator:
            user_features = None
            x = np.empty([self.T, self.n_arms, self.n_features])

            for i,line in enumerate(input_generator):
                logline = str(line).strip().split(" ")
                chosen = int(logline.pop(7))  # chosen article
                reward = int(logline.pop(7))  # 0 or 1
                #                     time = int(logline[0])  # timestamp
                user_features = [float(x) for x in logline[1:7]]

                for j, article in enumerate(self.article_keys):
                    article_feature = self.articles[article]
                    combinedFeature = self.articles[article] + user_features
                    x[i][j] = np.array(combinedFeature)
  
        self.features = x

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        x = np.empty([self.T, self.n_arms])
        with io.open(self.log_file, 'rb', buffering=1024 * 1024 * 512) as input_generator:
            for i,line in enumerate(input_generator):
                reward_array = np.array([-1 for i in range(self.n_arms)])
                logline = str(line).strip().split(" ")
                chosen = int(logline.pop(7))  # chosen article
                reward = int(logline.pop(7))  # 0 or 1

                reward_array[self.article_keys.index(chosen)] = reward
                x[i] = reward_array

        self.rewards = x
        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
