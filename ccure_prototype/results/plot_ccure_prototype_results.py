"""
Plots a bunch of results for C-Cure prototype runs

@author Dennis Soemers
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from math import ceil

# configuration for which we wish to plot results
config_to_plot = 1

# list of all the seeds for which we wish to plot results. empty list = all seeds
seeds_to_plot = [
    #1923542738,
    #2059921134,
    #1730383209,
    #15876390,
    #1398548544,
    #1484293880,
    #1631605134,
    #88742031,
    #384378282,
    #981629647,
    #178216022,
    #1940194933,
]

#RL_agent_to_plot = "AlwaysAuthenticateAgent"
RL_agent_to_plot = "ConcurrentTrueOnlineSarsaLambda"
#RL_agent_to_plot = "ConcurrentTrueOnlineSarsaLambda_06"
#RL_agent_to_plot = "ConcurrentTrueOnlineSarsaLambda_07"
#RL_agent_to_plot = "ConcurrentTrueOnlineSarsaLambda_09"
#RL_agent_to_plot = "NeverAuthenticateAgent"
#RL_agent_to_plot = "NStepSarsa_1"
#RL_agent_to_plot = "NStepSarsa_2"
#RL_agent_to_plot = "NStepSarsa_4"
#RL_agent_to_plot = "NStepSarsa_8"
#RL_agent_to_plot = "OracleAgent"
#RL_agent_to_plot = "Old_ConcurrentTrueOnlineSarsaLambda"
#RL_agent_to_plot = "RandomAgent"
#RL_agent_to_plot = "Sarsa"

# filenames (without .csv extension) for which we want to create subplots
files_to_plot = [
    'cumulative_rewards',
    'num_transactions',
    'num_genuines',
    'num_frauds',
    'num_secondary_auths',
    'num_secondary_auths_genuine',
    'num_secondary_auths_blocked_genuine',
    'num_secondary_auths_fraudulent',
    'total_population',
    'genuine_population',
    'fraud_population',
    'total_fraud_amounts_seen',
]

# tuples of filenames (without .csv extensions) for which we want to create percentage-plots where one file is
# "divided" by another file. Third element of tuple is title for subfigure
percentage_files_to_plot = [
    ('num_secondary_auths_genuine', 'num_genuines', 'Percentage Secondary Authentications Among Genuine Transactions'),
    ('num_secondary_auths_fraudulent', 'num_frauds', 'Percentage Secondary Authentications Among Fraudulent Transactions'),
]

# filenames (without .csv extension) for which we want to create subplots containing all models
files_to_plot_per_model = [
    'num_true_positives',
    'num_false_positives',
    'num_true_negatives',
    'num_false_negatives',
    'total_fraud_amounts_detected',
    'num_agreements_all',
    'num_agreements_true_positive',
    'num_agreements_false_positive',
    'num_agreements_true_negative',
    'num_agreements_false_negative',
]

plot_q_values = False

plot_rl_weights = False

plot_secondary_auth_percentages = False

plot_per_model_bins = False

bin_percentage_files_to_plot_per_model = [
    ('perc_true_positive_bins', 'True Positives'),
    ('perc_false_positive_bins', 'False Positives'),
    ('perc_true_negative_bins', 'True Negatives'),
    ('perc_false_positive_bins', 'False Negatives'),
    ('perc_agreements_true_positive_bins', 'True Positive Agreements'),
    ('perc_agreements_false_positive_bins', 'False Positive Agreements'),
    ('perc_agreements_true_negative_bins', 'True Negative Agreements'),
    ('perc_agreements_false_negative_bins', 'False Negative Agreements')
]


def create_subfigure(fig, filename, num_cols, num_rows, subfigure_idx, run_dirs):
    timesteps = []
    sum_results = []

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "{}.csv".format(filename))) as results_file:
            lines = results_file.readlines()

            if len(lines) == 0:
                print("0 lines in file: {}".format(os.path.join(run_dir, "{}.csv".format(filename))))

            if len(lines) < len(timesteps):
                # don't have as many results in this file as in some other, so need to discard some at the end
                timesteps = timesteps[:len(lines)]
                sum_results = sum_results[:len(lines)]

            if 0 < len(timesteps) < len(lines):
                # have more results in this new file than in some other file, so discard some of these new results
                lines = lines[:len(timesteps)]

            for line in lines:
                t, result = line.rstrip('\n').split(", ")
                t = int(t)
                result = float(result)

                if t == len(timesteps):
                    timesteps.append(t)
                    sum_results.append(result)
                else:
                    sum_results[t] += result

    mean_results = [s / len(run_dirs) for s in sum_results]

    ax = fig.add_subplot(num_rows, num_cols, subfigure_idx)
    ax.set_title(filename)
    ax.plot(timesteps, mean_results)
    ax.grid(color='k', linestyle='dotted')


def create_percentage_subfigure(fig, filename_1, filename_2, title, num_cols, num_rows, subfigure_idx, run_dirs):
    timesteps = []
    sum_results_1 = []

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "{}.csv".format(filename_1))) as results_file:
            lines = results_file.readlines()

            if len(lines) < len(timesteps):
                # don't have as many results in this file as in some other, so need to discard some at the end
                timesteps = timesteps[:len(lines)]
                sum_results_1 = sum_results_1[:len(lines)]

            if 0 < len(timesteps) < len(lines):
                # have more results in this new file than in some other file, so discard some of these new results
                lines = lines[:len(timesteps)]

            for line in lines:
                t, result = line.rstrip('\n').split(", ")
                t = int(t)
                result = float(result)

                if t == len(timesteps):
                    timesteps.append(t)
                    sum_results_1.append(result)
                else:
                    sum_results_1[t] += result

    mean_results_1 = np.asarray([s / len(run_dirs) for s in sum_results_1])

    sum_results_2 = [0] * len(timesteps)

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "{}.csv".format(filename_2))) as results_file:
            lines = results_file.readlines()

            if len(lines) < len(timesteps):
                # don't have as many results in this file as in some other, so need to discard some at the end
                timesteps = timesteps[:len(lines)]
                sum_results_2 = sum_results_2[:len(lines)]

            if 0 < len(timesteps) < len(lines):
                # have more results in this new file than in some other file, so discard some of these new results
                lines = lines[:len(timesteps)]

            for line in lines:
                t, result = line.rstrip('\n').split(", ")
                t = int(t)
                result = float(result)

                if t == len(timesteps):
                    timesteps.append(t)
                    sum_results_2.append(result)
                else:
                    sum_results_2[t] += result

    mean_results_2 = np.asarray([s / len(run_dirs) for s in sum_results_2])

    # don't want to divide by 0:
    mean_results_2[mean_results_2 == 0] = 1

    ax = fig.add_subplot(num_rows, num_cols, subfigure_idx)
    ax.set_title(title)
    ax.plot(timesteps, mean_results_1 / mean_results_2)
    ax.grid(color='k', linestyle='dotted')
    ax.set_ylim([0.0, 1.0])


def create_per_model_subfigure(fig, filename, num_cols, num_rows, subfigure_idx, run_dirs):
    timesteps = []
    sum_results_per_model = {}

    for run_dir in run_dirs:
        run_files = os.listdir(run_dir)
        per_model_files = [run_file for run_file in run_files if run_file.startswith(filename)]

        for per_model_file in per_model_files:
            model_name = per_model_file[len(filename)+1:-len(".csv")]

            if model_name not in sum_results_per_model:
                sum_results_per_model[model_name] = []

            sum_results = sum_results_per_model[model_name]

            with open(os.path.join(run_dir, per_model_file)) as results_file:
                lines = results_file.readlines()

                if len(lines) < len(timesteps):
                    # don't have as many results in this file as in some other, so need to discard some at the end
                    timesteps = timesteps[:len(lines)]
                    sum_results = sum_results[:len(lines)]

                if 0 < len(timesteps) < len(lines):
                    # have more results in this new file than in some other file, so discard some of these new results
                    lines = lines[:len(timesteps)]

                for line in lines:
                    t, result = line.rstrip('\n').split(", ")
                    t = int(t)
                    result = float(result)

                    if t == len(timesteps):
                        timesteps.append(t)

                    if t == len(sum_results):
                        sum_results.append(result)
                    else:
                        sum_results[t] += result

    ax = fig.add_subplot(num_rows, num_cols, subfigure_idx)
    ax.set_title(filename)

    for model_name in sum_results_per_model:
        mean_results = [s / len(run_dirs) for s in sum_results_per_model[model_name]]

        ax.plot(timesteps, mean_results, label=model_name)

    ax.grid(color='k', linestyle='dotted')


if __name__ == '__main__':
    sns.set()
    sns.set_style(style='white')

    total_num_subplots = len(files_to_plot)
    num_subfigure_cols = min(4, total_num_subplots)
    num_subfigure_rows = ceil(total_num_subplots / num_subfigure_cols)

    results_dir = os.path.dirname(__file__)

    config_dir = os.path.join(results_dir, "config{0:05d}_dir".format(config_to_plot))

    if len(seeds_to_plot) == 0:
        seed_dirs = [os.path.join(config_dir, seed_dir)
                     for seed_dir in os.listdir(config_dir) if seed_dir.startswith('seed_')]
    else:
        seed_dirs = [os.path.join(config_dir, 'seed_{}'.format(seed)) for seed in seeds_to_plot]

    rl_agent_dirs = [os.path.join(seed_dir, RL_agent_to_plot) for seed_dir in seed_dirs]

    run_dirs = []

    for rl_agent_dir in rl_agent_dirs:
        if os.path.isdir(rl_agent_dir):
            run_dirs.extend([os.path.join(rl_agent_dir, run_dir)
                             for run_dir in os.listdir(rl_agent_dir) if run_dir.startswith('run')])

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("C-Cure Prototype Results - {}".format(RL_agent_to_plot))

    subfigure_idx = 1

    for filename in files_to_plot:
        create_subfigure(fig=fig, filename=filename, num_cols=num_subfigure_cols, num_rows=num_subfigure_rows,
                         subfigure_idx=subfigure_idx, run_dirs=run_dirs)
        subfigure_idx += 1

    # -----------------------------------------------------------------------------------------------------------

    total_num_subplots = len(percentage_files_to_plot)
    num_subfigure_cols = min(4, total_num_subplots)
    num_subfigure_rows = ceil(total_num_subplots / num_subfigure_cols)

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("C-Cure Prototype Results - Sec. Auth. Percentages - {}".format(RL_agent_to_plot))

    subfigure_idx = 1

    for (filename_1, filename_2, title) in percentage_files_to_plot:
        create_percentage_subfigure(fig=fig, filename_1=filename_1, filename_2=filename_2, title=title,
                                    num_cols=num_subfigure_cols, num_rows=num_subfigure_rows,
                                    subfigure_idx=subfigure_idx, run_dirs=run_dirs)
        subfigure_idx += 1

    # -----------------------------------------------------------------------------------------------------------

    total_num_subplots = len(files_to_plot_per_model)
    num_subfigure_cols = min(4, total_num_subplots)
    num_subfigure_rows = ceil(total_num_subplots / num_subfigure_cols)

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("C-Cure Prototype Results - {} - Cost-Sensitive Models".format(RL_agent_to_plot))

    subfigure_idx = 1

    for filename in files_to_plot_per_model:
        create_per_model_subfigure(fig=fig, filename=filename, num_cols=num_subfigure_cols, num_rows=num_subfigure_rows,
                                   subfigure_idx=subfigure_idx, run_dirs=run_dirs)
        subfigure_idx += 1

    plt.legend(loc=2, fontsize=15, frameon=True).draggable()

    # -----------------------------------------------------------------------------------------------------------

    if plot_q_values:
        total_num_subplots = 4
        num_subfigure_cols = 2
        num_subfigure_rows = 2

        fig = plt.figure(figsize=(18, 9))
        fig.suptitle("C-Cure Prototype Results - {} - Q-Values".format(RL_agent_to_plot))

        create_subfigure(fig=fig, filename='q_values_genuine_no_auth', num_cols=num_subfigure_cols,
                         num_rows=num_subfigure_rows, subfigure_idx=1, run_dirs=run_dirs)

        create_subfigure(fig=fig, filename='q_values_genuine_auth', num_cols=num_subfigure_cols,
                         num_rows=num_subfigure_rows, subfigure_idx=2, run_dirs=run_dirs)

        create_subfigure(fig=fig, filename='q_values_fraud_no_auth', num_cols=num_subfigure_cols,
                         num_rows=num_subfigure_rows, subfigure_idx=3, run_dirs=run_dirs)

        create_subfigure(fig=fig, filename='q_values_fraud_auth', num_cols=num_subfigure_cols,
                         num_rows=num_subfigure_rows, subfigure_idx=4, run_dirs=run_dirs)

    # -----------------------------------------------------------------------------------------------------------

    if plot_rl_weights:
        if RL_agent_to_plot == "TrueOnlineSarsaLambda":
            num_actions = 2
        else:
            num_actions = 3

        num_weights = 11

        subfigure_idx = 1

        fig = plt.figure(figsize=(18, 9))
        #fig.suptitle("C-Cure Prototype Results - {} - RL Weights".format(RL_agent_to_plot))

        for action in range(num_actions):
            for weight in range(num_weights):
                create_subfigure(fig=fig, filename="action_{}_weight_{}".format(action, weight), num_cols=num_weights,
                                 num_rows=num_actions, subfigure_idx=subfigure_idx, run_dirs=run_dirs)
                subfigure_idx += 1

    plt.tight_layout()

    # -----------------------------------------------------------------------------------------------------------

    if plot_secondary_auth_percentages:
        fig = plt.figure(figsize=(18, 9))
        fig.suptitle("C-Cure Prototype Results - {} - Percentage Secondary Authentications per Transaction Counter".format(RL_agent_to_plot))

        transaction_counters = []
        file_occurrences = []
        sum_results = []

        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "secondary_auth_percentages.csv")) as file:
                lines = file.readlines()

                for line in lines:
                    transaction_counter, ratio = line.rstrip('\n').split(", ")
                    transaction_counter = int(transaction_counter)
                    percentage = float(ratio) * 100.0

                    while transaction_counter >= len(transaction_counters):
                        transaction_counters.append(0)
                        file_occurrences.append(0)
                        sum_results.append(0.0)

                    transaction_counters[transaction_counter] = transaction_counter
                    file_occurrences[transaction_counter] += 1
                    sum_results[transaction_counter] += percentage

        # get rid of the 0 index, there is no such thing as a 0th transaction, first is 1st
        transaction_counters = transaction_counters[1:]
        file_occurrences = file_occurrences[1:]
        sum_results = sum_results[1:]

        mean_results = [sum_results[i] / file_occurrences[i] for i in range(len(sum_results))]

        plt.bar(transaction_counters, mean_results)
        plt.grid(color='k', linestyle='dotted')

    # -----------------------------------------------------------------------------------------------------------

    if plot_per_model_bins:
        total_num_subplots = len(bin_percentage_files_to_plot_per_model)
        num_subfigure_cols = min(4, total_num_subplots)
        num_subfigure_rows = ceil(total_num_subplots / num_subfigure_cols)

        fig = plt.figure(figsize=(18, 9))
        fig.suptitle("C-Cure Prototype Results - {} - Model Performance per Transaction Counter".format(RL_agent_to_plot))

        subfigure_idx = 1

        for (filename, title) in bin_percentage_files_to_plot_per_model:
            transaction_counters = []
            file_occurrences_per_model = {}
            sum_results_per_model = {}

            for run_dir in run_dirs:
                run_files = os.listdir(run_dir)
                per_model_files = [run_file for run_file in run_files if run_file.startswith(filename)]

                for per_model_file in per_model_files:
                    model_name = per_model_file[len(filename) + 1:-len(".csv")]

                    if model_name not in sum_results_per_model:
                        file_occurrences_per_model[model_name] = [0] * len(transaction_counters)
                        sum_results_per_model[model_name] = [0] * len(transaction_counters)

                    sum_results = sum_results_per_model[model_name]

                    with open(os.path.join(run_dir, per_model_file)) as results_file:
                        lines = results_file.readlines()

                        for line in lines:
                            transaction_counter, ratio = line.rstrip('\n').split(", ")
                            transaction_counter = int(transaction_counter)
                            percentage = float(ratio) * 100.0

                            while transaction_counter >= len(transaction_counters):
                                transaction_counters.append(0)

                                for name in sum_results_per_model:
                                    file_occurrences_per_model[name].append(0)
                                    sum_results_per_model[name].append(0.0)

                            transaction_counters[transaction_counter] = transaction_counter
                            file_occurrences_per_model[model_name][transaction_counter] += 1
                            sum_results_per_model[model_name][transaction_counter] += percentage

            # get rid of the 0 index, there is no such thing as a 0th transaction, first is 1st
            transaction_counters = transaction_counters[1:]

            for model_name in sum_results_per_model:
                file_occurrences_per_model[model_name] = file_occurrences_per_model[model_name][1:]
                sum_results_per_model[model_name] = sum_results_per_model[model_name][1:]

            mean_results_per_model = {
                model_name: [sum_results_per_model[model_name][i] / file_occurrences_per_model[model_name][i]
                             for i in range(len(sum_results_per_model[model_name]))]
                for model_name in sum_results_per_model
            }

            model_names = [model_name for model_name in sum_results_per_model]
            total_bars_occupied_width = 0.9
            width_per_bar = total_bars_occupied_width / len(model_names)

            ax = fig.add_subplot(num_subfigure_rows, num_subfigure_cols, subfigure_idx)
            ax.set_title(title)

            xs = np.asarray(transaction_counters)
            ys = []

            for model_name in model_names:
                mean_results = [
                    sum_results_per_model[model_name][i] / file_occurrences_per_model[model_name][i]
                    for i in range(len(sum_results_per_model[model_name]))
                ]

                ys.append(mean_results)

            if len(model_names) % 2 == 0:
                # even number of models
                bar_offsets = [width_per_bar * (i - 0.5 * len(model_names)) for i in range(len(model_names))]
                align = 'edge'
            else:
                # odd number of models
                bar_offsets = [width_per_bar * (i - 0.5 * (len(model_names) - 1)) for i in range(len(model_names))]
                align = 'center'

            for i in range(len(model_names)):
                ax.bar(xs + bar_offsets[i], ys[i], label=model_names[i], align=align, width=width_per_bar)

            ax.grid(color='k', linestyle='dotted')

            subfigure_idx += 1

        plt.legend(loc=2, fontsize=15, frameon=True).draggable()

    plt.show()
