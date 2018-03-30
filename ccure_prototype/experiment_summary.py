"""
This module implements a class that can be used to summarize all kinds of data (e.g. rewards)
for experiments

@author Dennis Soemers
"""

import os


class ExperimentSummary:

    def __init__(self, flat_fee, relative_fee, output_dir, cs_model_config_names, write_frequency=200):
        self.cs_model_config_names = cs_model_config_names

        self.timesteps = [0, ]
        self.cumulative_rewards = [0.0, ]
        self.num_transactions = [0, ]
        self.num_genuines = [0, ]
        self.num_frauds = [0, ]
        self.num_secondary_auths = [0, ]
        self.num_secondary_auths_genuine = [0, ]
        self.num_secondary_auths_blocked_genuine = [0, ]
        self.num_secondary_auths_fraudulent = [0, ]

        self.total_population = [0, ]
        self.genuine_population = [0, ]
        self.fraud_population = [0, ]

        self.total_fraud_amounts_seen = [0.0, ]

        self.num_true_positives_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_false_positives_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_true_negatives_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_false_negatives_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.total_fraud_amounts_detected_per_model = {model_name: [0.0, ] for model_name in cs_model_config_names}
        self.num_agreements_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_agreements_true_positive_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_agreements_false_positive_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_agreements_true_negative_per_model = {model_name: [0, ] for model_name in cs_model_config_names}
        self.num_agreements_false_negative_per_model = {model_name: [0, ] for model_name in cs_model_config_names}

        self.cumulative_rewards_filepath = os.path.join(output_dir, "cumulative_rewards.csv")
        self.num_transactions_filepath = os.path.join(output_dir, "num_transactions.csv")
        self.num_genuines_filepath = os.path.join(output_dir, "num_genuines.csv")
        self.num_frauds_filepath = os.path.join(output_dir, "num_frauds.csv")
        self.num_secondary_auths_filepath = os.path.join(output_dir, "num_secondary_auths.csv")
        self.num_secondary_auths_genuine_filepath = os.path.join(output_dir, "num_secondary_auths_genuine.csv")
        self.num_secondary_auths_blocked_genuine_filepath = os.path.join(output_dir,
                                                                         "num_secondary_auths_blocked_genuine.csv")
        self.num_secondary_auths_fraudulent_filepath = os.path.join(output_dir, "num_secondary_auths_fraudulent.csv")

        self.total_population_filepath = os.path.join(output_dir, "total_population.csv")
        self.genuine_population_filepath = os.path.join(output_dir, "genuine_population.csv")
        self.fraud_population_filepath = os.path.join(output_dir, "fraud_population.csv")

        self.total_fraud_amounts_seen_filepath = os.path.join(output_dir, "total_fraud_amounts_seen.csv")

        self.num_true_positives_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_true_positives_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_false_positives_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_false_positives_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_true_negatives_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_true_negatives_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_false_negatives_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_false_negatives_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.total_fraud_amounts_detected_per_model_filepaths = {
            model_name: os.path.join(output_dir, "total_fraud_amounts_detected_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_agreements_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_agreements_all_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_agreements_true_positive_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_agreements_true_positive_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_agreements_false_positive_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_agreements_false_positive_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_agreements_true_negative_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_agreements_true_negative_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}
        self.num_agreements_false_negative_per_model_filepaths = {
            model_name: os.path.join(output_dir, "num_agreements_false_negative_{}.csv".format(
                model_name.replace(":", "_").replace("/", "_")))
            for model_name in cs_model_config_names}

        self.flat_fee = flat_fee
        self.relative_fee = relative_fee
        self.write_frequency = write_frequency

    def __enter__(self):
        self.cumulative_rewards_file = open(self.cumulative_rewards_filepath, 'x')
        self.num_transactions_file = open(self.num_transactions_filepath, 'x')
        self.num_genuines_file = open(self.num_genuines_filepath, 'x')
        self.num_frauds_file = open(self.num_frauds_filepath, 'x')
        self.num_secondary_auths_file = open(self.num_secondary_auths_filepath, 'x')
        self.num_secondary_auths_genuine_file = open(self.num_secondary_auths_genuine_filepath, 'x')
        self.num_secondary_auths_blocked_genuine_file = open(self.num_secondary_auths_blocked_genuine_filepath, 'x')
        self.num_secondary_auths_fraudulent_file = open(self.num_secondary_auths_fraudulent_filepath, 'x')

        self.total_population_file = open(self.total_population_filepath, 'x')
        self.genuine_population_file = open(self.genuine_population_filepath, 'x')
        self.fraud_population_file = open(self.fraud_population_filepath, 'x')

        self.total_fraud_amounts_seen_file = open(self.total_fraud_amounts_seen_filepath, 'x')

        self.num_true_positives_per_model_files = \
            {model_name: open(self.num_true_positives_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_false_positives_per_model_files = \
            {model_name: open(self.num_false_positives_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_true_negatives_per_model_files = \
            {model_name: open(self.num_true_negatives_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_false_negatives_per_model_files = \
            {model_name: open(self.num_false_negatives_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.total_fraud_amounts_detected_per_model_files = \
            {model_name: open(self.total_fraud_amounts_detected_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_agreements_per_model_files = \
            {model_name: open(self.num_agreements_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_agreements_true_positive_per_model_files = \
            {model_name: open(self.num_agreements_true_positive_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_agreements_false_positive_per_model_files = \
            {model_name: open(self.num_agreements_false_positive_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_agreements_true_negative_per_model_files = \
            {model_name: open(self.num_agreements_true_negative_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}
        self.num_agreements_false_negative_per_model_files = \
            {model_name: open(self.num_agreements_false_negative_per_model_filepaths[model_name], 'x')
             for model_name in self.cs_model_config_names}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write_output()

        self.cumulative_rewards_file.close()
        self.num_transactions_file.close()
        self.num_genuines_file.close()
        self.num_frauds_file.close()
        self.num_secondary_auths_file.close()
        self.num_secondary_auths_genuine_file.close()
        self.num_secondary_auths_blocked_genuine_file.close()
        self.num_secondary_auths_fraudulent_file.close()

        self.total_population_file.close()
        self.genuine_population_file.close()
        self.fraud_population_file.close()

        self.total_fraud_amounts_seen_file.close()

        for model_name in self.cs_model_config_names:
            self.num_true_positives_per_model_files[model_name].close()
            self.num_false_positives_per_model_files[model_name].close()
            self.num_true_negatives_per_model_files[model_name].close()
            self.num_false_negatives_per_model_files[model_name].close()
            self.total_fraud_amounts_detected_per_model_files[model_name].close()
            self.num_agreements_per_model_files[model_name].close()
            self.num_agreements_true_positive_per_model_files[model_name].close()
            self.num_agreements_false_positive_per_model_files[model_name].close()
            self.num_agreements_true_negative_per_model_files[model_name].close()
            self.num_agreements_false_negative_per_model_files[model_name].close()

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

        self.total_population.append(self.total_population[-1])
        self.genuine_population.append(self.genuine_population[-1])
        self.fraud_population.append(self.fraud_population[-1])

        self.total_fraud_amounts_seen.append(self.total_fraud_amounts_seen[-1])

        for model_name in self.cs_model_config_names:
            self.num_true_positives_per_model[model_name].append(self.num_true_positives_per_model[model_name][-1])
            self.num_false_positives_per_model[model_name].append(self.num_false_positives_per_model[model_name][-1])
            self.num_true_negatives_per_model[model_name].append(self.num_true_negatives_per_model[model_name][-1])
            self.num_false_negatives_per_model[model_name].append(self.num_false_negatives_per_model[model_name][-1])
            self.total_fraud_amounts_detected_per_model[model_name].append(
                self.total_fraud_amounts_detected_per_model[model_name][-1])
            self.num_agreements_per_model[model_name].append(self.num_agreements_per_model[model_name][-1])
            self.num_agreements_true_positive_per_model[model_name].append(
                self.num_agreements_true_positive_per_model[model_name][-1])
            self.num_agreements_false_positive_per_model[model_name].append(
                self.num_agreements_false_positive_per_model[model_name][-1])
            self.num_agreements_true_negative_per_model[model_name].append(
                self.num_agreements_true_negative_per_model[model_name][-1])
            self.num_agreements_false_negative_per_model[model_name].append(
                self.num_agreements_false_negative_per_model[model_name][-1])

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
        self.num_transactions_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_transactions[i + 1]) for i in range(len(self.num_transactions) - 1))
        self.num_genuines_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_genuines[i + 1]) for i in range(len(self.num_genuines) - 1))
        self.num_frauds_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_frauds[i + 1]) for i in range(len(self.num_frauds) - 1))
        self.num_secondary_auths_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_secondary_auths[i + 1]) for i in range(len(self.num_secondary_auths) - 1))
        self.num_secondary_auths_genuine_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_secondary_auths_genuine[i + 1]) for i in range(len(self.num_secondary_auths_genuine) - 1))
        self.num_secondary_auths_blocked_genuine_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_secondary_auths_blocked_genuine[i + 1]) for i in range(len(self.num_secondary_auths_blocked_genuine) - 1))
        self.num_secondary_auths_fraudulent_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.num_secondary_auths_fraudulent[i + 1]) for i in range(len(self.num_secondary_auths_fraudulent) - 1))

        self.total_population_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.total_population[i + 1]) for i in range(len(self.total_population) - 1))
        self.genuine_population_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.genuine_population[i + 1]) for i in range(len(self.genuine_population) - 1))
        self.fraud_population_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.fraud_population[i + 1]) for i in range(len(self.fraud_population) - 1))

        self.total_fraud_amounts_seen_file.writelines("{}, {}\n".format(
            self.timesteps[i + 1], self.total_fraud_amounts_seen[i + 1]) for i in range(len(self.total_fraud_amounts_seen) - 1))

        for model_name in self.cs_model_config_names:
            self.num_true_positives_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_true_positives_per_model[model_name][i + 1]) for i in range(
                len(self.num_true_positives_per_model[model_name]) - 1))
            self.num_false_positives_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_false_positives_per_model[model_name][i + 1]) for i in range(
                len(self.num_false_positives_per_model[model_name]) - 1))
            self.num_true_negatives_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_true_negatives_per_model[model_name][i + 1]) for i in range(
                len(self.num_true_negatives_per_model[model_name]) - 1))
            self.num_false_negatives_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_false_negatives_per_model[model_name][i + 1]) for i in range(
                len(self.num_false_negatives_per_model[model_name]) - 1))
            self.total_fraud_amounts_detected_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.total_fraud_amounts_detected_per_model[model_name][i + 1]) for i in range(
                len(self.total_fraud_amounts_detected_per_model[model_name]) - 1))
            self.num_agreements_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_agreements_per_model[model_name][i + 1]) for i in range(
                len(self.num_agreements_per_model[model_name]) - 1))
            self.num_agreements_true_positive_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_agreements_true_positive_per_model[model_name][i + 1]) for i in range(
                len(self.num_agreements_true_positive_per_model[model_name]) - 1))
            self.num_agreements_false_positive_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_agreements_false_positive_per_model[model_name][i + 1]) for i in range(
                len(self.num_agreements_false_positive_per_model[model_name]) - 1))
            self.num_agreements_true_negative_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_agreements_true_negative_per_model[model_name][i + 1]) for i in range(
                len(self.num_agreements_true_negative_per_model[model_name]) - 1))
            self.num_agreements_false_negative_per_model_files[model_name].writelines("{}, {}\n".format(
                self.timesteps[i + 1], self.num_agreements_false_negative_per_model[model_name][i + 1]) for i in range(
                len(self.num_agreements_false_negative_per_model[model_name]) - 1))

        self.timesteps = [self.timesteps[-1], ]
        self.cumulative_rewards = [self.cumulative_rewards[-1], ]
        self.num_transactions = [self.num_transactions[-1], ]
        self.num_genuines = [self.num_genuines[-1], ]
        self.num_frauds = [self.num_frauds[-1], ]
        self.num_secondary_auths = [self.num_secondary_auths[-1], ]
        self.num_secondary_auths_genuine = [self.num_secondary_auths_genuine[-1], ]
        self.num_secondary_auths_blocked_genuine = [self.num_secondary_auths_blocked_genuine[-1], ]
        self.num_secondary_auths_fraudulent = [self.num_secondary_auths_fraudulent[-1], ]

        self.total_population = [self.total_population[-1], ]
        self.genuine_population = [self.genuine_population[-1], ]
        self.fraud_population = [self.fraud_population[-1], ]

        self.total_fraud_amounts_seen = [self.total_fraud_amounts_seen[-1], ]

        for model_name in self.cs_model_config_names:
            self.num_true_positives_per_model[model_name] = [self.num_true_positives_per_model[model_name][-1], ]
            self.num_false_positives_per_model[model_name] = [self.num_false_positives_per_model[model_name][-1], ]
            self.num_true_negatives_per_model[model_name] = [self.num_true_negatives_per_model[model_name][-1], ]
            self.num_false_negatives_per_model[model_name] = [self.num_false_negatives_per_model[model_name][-1], ]
            self.total_fraud_amounts_detected_per_model[model_name] = \
                [self.total_fraud_amounts_detected_per_model[model_name][-1], ]
            self.num_agreements_per_model[model_name] = [self.num_agreements_per_model[model_name][-1], ]
            self.num_agreements_true_positive_per_model[model_name] = \
                [self.num_agreements_true_positive_per_model[model_name][-1], ]
            self.num_agreements_false_positive_per_model[model_name] = \
                [self.num_agreements_false_positive_per_model[model_name][-1], ]
            self.num_agreements_true_negative_per_model[model_name] = \
                [self.num_agreements_true_negative_per_model[model_name][-1], ]
            self.num_agreements_false_negative_per_model[model_name] = \
                [self.num_agreements_false_negative_per_model[model_name][-1], ]
