"""
Plots a bunch of results for C-Cure prototype runs

@author Dennis Soemers
"""

import matplotlib.pyplot as plt
import os
import seaborn as sns
from math import ceil

# configuration for which we wish to plot results
config_to_plot = 0

# list of all the seeds for which we wish to plot results. empty list = all seeds
seeds_to_plot = [
    #1406165082,
    357552348,
]

#RL_agent_to_plot = "TrueOnlineSarsaLambda"
RL_agent_to_plot = "ConcurrentTrueOnlineSarsaLambda"

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

plot_q_values = True

plot_rl_weights = True


def create_subfigure(fig, filename, num_cols, num_rows, subfigure_idx, run_dirs):
    timesteps = []
    sum_results = []

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "{}.csv".format(filename))) as results_file:
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
                    sum_results.append(result)
                else:
                    sum_results[t] += result

    mean_results = [s / len(run_dirs) for s in sum_results]

    ax = fig.add_subplot(num_rows, num_cols, subfigure_idx)
    ax.set_title(filename)
    ax.plot(timesteps, mean_results)
    ax.grid(color='k', linestyle='dotted')


def create_per_model_subfigure(fig, filename, num_cols, num_rows, subfigure_idx, run_dirs):
    timesteps = []
    sum_results_per_model = {}

    for run_dir in run_dirs:
        run_files = os.listdir(run_dir)
        per_model_files = [run_file for run_file in run_files if run_file.startswith(filename)]

        for per_model_file in per_model_files:
            model_name = per_model_file[len(filename)+1:-len(".csv")]        # TODO probably +2 instead of +1?

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

    run_dirs = [os.path.join(rl_agent_dir, run_dir)
                for rl_agent_dir in rl_agent_dirs for run_dir in os.listdir(rl_agent_dir) if run_dir.startswith('run')]

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("C-Cure Prototype Results - {}".format(RL_agent_to_plot))

    subfigure_idx = 1

    for filename in files_to_plot:
        create_subfigure(fig=fig, filename=filename, num_cols=num_subfigure_cols, num_rows=num_subfigure_rows,
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
    plt.show()
