"""
This module implements a class that can be used to summarize all kinds of data (e.g. rewards)
for experiments

@author Dennis Soemers
"""


class ExperimentSummary:

    def __init__(self, flat_fee, relative_fee, cumulative_rewards_filepath, write_frequency=200):
        self.timesteps = [0, ]
        self.cumulative_rewards = [0, ]
        self.num_transactions = [0, ]
        self.num_genuines = [0, ]
        self.num_frauds = [0, ]
        self.num_secondary_auths = [0, ]
        self.num_secondary_auths_genuine = [0, ]
        self.num_secondary_auths_blocked_genuine = [0, ]
        self.num_secondary_auths_fraudulent = [0, ]

        self.cumulative_rewards_filepath = cumulative_rewards_filepath

        self.flat_fee = flat_fee
        self.relative_fee = relative_fee
        self.write_frequency = write_frequency

    def __enter__(self):
        self.cumulative_rewards_file = open(self.cumulative_rewards_filepath, 'x')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write_output()

        self.cumulative_rewards_file.close()

        return exc_val is None

    def new_timestep(self, t):
        if len(self.timesteps) % self.write_frequency == 0:
            self.write_output()

        self.timesteps.append(t)

        self.cumulative_rewards.append(self.cumulative_rewards[-1])
        self.num_transactions.append(self.num_transactions[-1])
        self.num_genuines.append(self.num_genuines[-1])
        self.num_frauds.append(self.num_frauds[-1])
        self.num_secondary_auths.append(self.num_secondary_auths[-1])
        self.num_secondary_auths_genuine.append(self.num_secondary_auths_genuine[-1])
        self.num_secondary_auths_blocked_genuine.append(self.num_secondary_auths_blocked_genuine[-1])
        self.num_secondary_auths_fraudulent.append(self.num_secondary_auths_fraudulent[-1])

    def record_transaction(self, transaction):
        self.num_transactions[-1] += 1

        if transaction.Target == 1:
            self.num_frauds[-1] += 1
            new_reward = - transaction.Amount
        else:
            self.num_genuines[-1] += 1
            new_reward = self.flat_fee + self.relative_fee * transaction.Amount

        self.cumulative_rewards[-1] += new_reward

    def write_output(self):
        self.cumulative_rewards_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.cumulative_rewards[i + 1]) for i in range(len(self.cumulative_rewards) - 1))

        self.timesteps = [self.timesteps[-1], ]
        self.cumulative_rewards = [self.cumulative_rewards[-1], ]
        self.num_transactions = [self.num_transactions[-1], ]
        self.num_genuines = [self.num_genuines[-1], ]
        self.num_frauds = [self.num_frauds[-1], ]
        self.num_secondary_auths = [self.num_secondary_auths[-1], ]
        self.num_secondary_auths_genuine = [self.num_secondary_auths_genuine[-1], ]
        self.num_secondary_auths_blocked_genuine = [self.num_secondary_auths_blocked_genuine[-1], ]
        self.num_secondary_auths_fraudulent = [self.num_secondary_auths_fraudulent[-1], ]
