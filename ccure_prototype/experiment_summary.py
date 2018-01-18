"""
This module implements a class that can be used to summarize all kinds of data (e.g. rewards)
for experiments

@author Dennis Soemers
"""


class ExperimentSummary:

    def __init__(self, flat_fee, relative_fee, cumulative_rewards_filepath, write_frequency=200):
        self.last_cumulative_reward = 0
        self.last_timestep = -1

        self.timesteps = []
        self.cumulative_rewards = []

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

    def record_transaction(self, transaction):
        if transaction.Target == 1:
            new_reward = - transaction.Amount
        else:
            new_reward = self.flat_fee + self.relative_fee * transaction.Amount

        self.last_cumulative_reward += new_reward
        self.last_timestep += 1

        self.timesteps.append(self.last_timestep)
        self.cumulative_rewards.append(self.last_cumulative_reward)

        if len(self.timesteps) % self.write_frequency == 0:
            self.write_output()

    def write_output(self):
        self.cumulative_rewards_file.writelines("{}, {}\n".format(
            self.timesteps[i], self.cumulative_rewards[i]) for i in range(len(self.cumulative_rewards)))

        self.cumulative_rewards = []
        self.timesteps = []
