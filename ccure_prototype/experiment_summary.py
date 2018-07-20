"""
This module implements a class that can be used to summarize all kinds of data (e.g. rewards)
for experiments

@author Dennis Soemers
"""

import os


class ExperimentSummary:

    def __init__(self, flat_fee, relative_fee, output_dir, cs_model_config_names,
                 rl_authenticator, rl_agent, write_frequency=200):

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

        self.num_actions = 3

        current_rl_weights = rl_agent.get_weights()
        self.num_weights_per_action = int(current_rl_weights.shape[0] / self.num_actions)

        self.weights_per_action = []
        for action in range(self.num_actions):
            self.weights_per_action.append([])

            for i in range(self.num_weights_per_action):
                self.weights_per_action[action].append([current_rl_weights[i + action * self.num_weights_per_action]])

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

        self.num_true_positive_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_false_positive_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_true_negative_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_false_negative_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_agreements_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_agreements_true_positive_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_agreements_false_positive_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_agreements_true_negative_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}
        self.num_agreements_false_negative_bins_per_model = {model_name: [[], ] for model_name in cs_model_config_names}

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

        self.genuine_q_values_no_auth_filepath = os.path.join(output_dir, "q_values_genuine_no_auth.csv")
        self.genuine_q_values_auth_filepath = os.path.join(output_dir, "q_values_genuine_auth.csv")
        self.fraud_q_values_no_auth_filepath = os.path.join(output_dir, "q_values_fraud_no_auth.csv")
        self.fraud_q_values_auth_filepath = os.path.join(output_dir, "q_values_fraud_auth.csv")

        self.secondary_auth_percentages_filepath = os.path.join(output_dir, "secondary_auth_percentages.csv")
        self.num_bin_entries_filepath = os.path.join(output_dir, "num_bin_entries.csv")

        self.output_dir = output_dir

        self.weight_filepaths_per_action = []
        for action in range(self.num_actions):
            self.weight_filepaths_per_action.append([])

            for i in range(self.num_weights_per_action):
                self.weight_filepaths_per_action[action].append(
                    os.path.join(output_dir, "action_{}_weight_{}.csv".format(action, i)))

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

        self.rl_authenticator = rl_authenticator
        self.rl_agent = rl_agent

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

        self.weight_files_per_action = []
        for action in range(self.num_actions):
            self.weight_files_per_action.append([])

            for i in range(self.num_weights_per_action):
                self.weight_files_per_action[action].append(open(self.weight_filepaths_per_action[action][i], 'x'))

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

        for action in range(self.num_actions):
            for i in range(self.num_weights_per_action):
                self.weight_files_per_action[action][i].close()

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

        # write Q values of RL agent
        if self.rl_agent is not None:
            with open(self.genuine_q_values_no_auth_filepath, 'x') as file:
                file.writelines("{}, {}\n".format(
                    i, self.rl_agent.genuine_q_values_no_auth[i]) for i in range(
                    len(self.rl_agent.genuine_q_values_no_auth)
                ))

            with open(self.genuine_q_values_auth_filepath, 'x') as file:
                file.writelines("{}, {}\n".format(
                    i, self.rl_agent.genuine_q_values_auth[i]) for i in range(
                    len(self.rl_agent.genuine_q_values_auth)
                ))

            with open(self.fraud_q_values_no_auth_filepath, 'x') as file:
                file.writelines("{}, {}\n".format(
                    i, self.rl_agent.fraud_q_values_no_auth[i]) for i in range(
                    len(self.rl_agent.fraud_q_values_no_auth)
                ))

            with open(self.fraud_q_values_auth_filepath, 'x') as file:
                file.writelines("{}, {}\n".format(
                    i, self.rl_agent.fraud_q_values_auth[i]) for i in range(
                    len(self.rl_agent.fraud_q_values_auth)
                ))

        # write data for bins of i'th transaction per customer
        if self.rl_authenticator is not None:
            with open(self.secondary_auth_percentages_filepath, 'x') as file:
                file.writelines("{}, {}\n".format(
                    i, self.rl_authenticator.second_auth_count_bins[i] / self.rl_authenticator.trans_count_bins[i])
                                for i in range(1, len(self.rl_authenticator.second_auth_count_bins))
                )

            with open(self.num_bin_entries_filepath, 'x') as file:
                file.writelines("{}, {}\n".format(
                    i, self.rl_authenticator.trans_count_bins[i])
                                for i in range(1, len(self.rl_authenticator.trans_count_bins))
                )

            # define helper function to get entries for bins (auto return 0 if bin doesn't exist)
            def get_bin_entry(bin_list, bin_idx):
                if bin_idx < len(bin_list):
                    return bin_list[bin_idx]
                else:
                    return 0

            trans_count_bins = self.rl_authenticator.trans_count_bins

            for model_name in self.cs_model_config_names:
                model_summ = self.rl_authenticator.cs_model_performance_summaries[model_name]

                with open(os.path.join(self.output_dir, "perc_true_positive_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_true_positive_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_false_positive_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_false_positive_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_true_negative_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_true_negative_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_false_negative_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_false_negative_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_agreements_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_agreements_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_agreements_true_positive_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_agreements_true_positive_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_agreements_false_positive_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_agreements_false_positive_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_agreements_true_negative_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_agreements_true_negative_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

                with open(os.path.join(self.output_dir, "perc_agreements_false_negative_bins_{}.csv".format(
                        model_name.replace(":", "_").replace("/", "_"))), 'x') as file:

                    file.writelines("{}, {}\n".format(
                        i, get_bin_entry(model_summ.num_agreements_false_negative_bins, i) / trans_count_bins[i])
                                    for i in range(1, len(self.rl_authenticator.trans_count_bins))
                    )

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

        for action in range(self.num_actions):
            for i in range(self.num_weights_per_action):
                self.weights_per_action[action][i].append(self.weights_per_action[action][i][-1])

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

        for action in range(self.num_actions):
            for w in range(self.num_weights_per_action):
                self.weight_files_per_action[action][w].writelines("{}, {}\n".format(
                    self.timesteps[i + 1], self.weights_per_action[action][w][i + 1]) for i in range(len(self.weights_per_action[action][w]) - 1))

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

        for action in range(self.num_actions):
            for w in range(self.num_weights_per_action):
                self.weights_per_action[action][w] = [self.weights_per_action[action][w][-1], ]

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
