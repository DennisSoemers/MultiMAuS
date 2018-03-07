"""
Plots a bunch of results for C-Cure prototype runs

@author Dennis Soemers
"""

import matplotlib.pyplot as plt
import os
import seaborn as sns

# list of all the seeds for which we wish to plot results. empty list = all seeds
seeds_to_plot = [
    1357973141,
]

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
]


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


if __name__ == '__main__':
    sns.set()
    sns.set_style(style='white')

    total_num_subplots = len(files_to_plot)
    num_subfigure_cols = min(4, total_num_subplots)
    num_subfigure_rows = (total_num_subplots + (total_num_subplots % num_subfigure_cols)) / num_subfigure_cols

    results_dir = os.path.dirname(__file__)

    if len(seeds_to_plot) == 0:
        seed_dirs = [os.path.join(results_dir, seed_dir)
                     for seed_dir in os.listdir(results_dir) if seed_dir.startswith('seed_')]
    else:
        seed_dirs = [os.path.join(results_dir, 'seed_{}'.format(seed)) for seed in seeds_to_plot]

    run_dirs = [os.path.join(seed_dir, run_dir) for seed_dir in seed_dirs for run_dir in os.listdir(seed_dir)]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("C-Cure Prototype Results")

    subfigure_idx = 1

    for filename in files_to_plot:
        create_subfigure(fig=fig, filename=filename, num_cols=num_subfigure_cols, num_rows=num_subfigure_rows,
                         subfigure_idx=subfigure_idx, run_dirs=run_dirs)
        subfigure_idx += 1

    # plt.legend(loc=2, fontsize=15, frameon=True).draggable()
    plt.show()
