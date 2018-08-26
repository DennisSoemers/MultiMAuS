"""

@author Dennis Soemers
"""

import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as st
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

plt.rc('text', usetex=True)

ALL_ALGORITHMS = [
    'ConcurrentTrueOnlineSarsaLambda_06',
    'ConcurrentTrueOnlineSarsaLambda_07',
    'ConcurrentTrueOnlineSarsaLambda',
    'ConcurrentTrueOnlineSarsaLambda_09',
    'NStepSarsa_1',
    'NStepSarsa_2',
    'NStepSarsa_4',
    'NStepSarsa_8',
    'AlwaysAuthenticateAgent',
    'NeverAuthenticateAgent',
    'RandomAgent',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figures showing performance on MultiMAuS simulator.')

    parser.add_argument('--algorithms', dest='algorithms', nargs='+', default=ALL_ALGORITHMS, choices=ALL_ALGORITHMS,
                        help='Algorithms for which to plot performance.')
    parser.add_argument('--config', dest='config', default=0, type=int,
                        help='MultiMAuS configuration for which to plot results.')
    parser.add_argument('--plot-conf-intervals', dest='plot_conf_intervals', action='store_true',
                        help='Plot 95% confidence intervals around mean results.')
    parser.add_argument('--results-dir', dest='results_dir', default="D:/Apps/MultiMAuS-fork/ccure_prototype/results",
                        help='Directory containing result files.')
    parser.add_argument('--out-dir', dest='out_dir', default=None,
                        help='Directory in which to save figures. Will overwrite if already existing.')

    parsed_args = parser.parse_args()
    #parsed_args.plot_conf_intervals = True

    # --------------------------------------------------------------------------------------------------------------

    sns.set()
    sns.set_style(style='white')

    blues_palette = itertools.cycle(sns.color_palette("Blues_r", 5))
    oranges_palette = itertools.cycle(sns.color_palette("Oranges_r", 5))
    greens_palette = itertools.cycle(sns.color_palette("Greens_r", 5))

    algorithms_plot_config = {
        'ConcurrentTrueOnlineSarsaLambda_06': {
            'friendly_name': r'concurrent true online sarsa($0.6$)',
            'color': next(blues_palette)
        },
        'ConcurrentTrueOnlineSarsaLambda_07': {
            'friendly_name': r'concurrent true online sarsa($0.7$)',
            'color': next(blues_palette)
        },
        'ConcurrentTrueOnlineSarsaLambda': {
            'friendly_name': r'concurrent true online sarsa($0.8$)',
            'color': next(blues_palette)
        },
        'ConcurrentTrueOnlineSarsaLambda_09': {
            'friendly_name': r'concurrent true online sarsa($0.9$)',
            'color': next(blues_palette)
        },
        'NStepSarsa_1': {
            'friendly_name': r'semi-gradient $1$-step sarsa',
            'color': next(oranges_palette)
        },
        'NStepSarsa_2': {
            'friendly_name': r'semi-gradient $2$-step sarsa',
            'color': next(oranges_palette)
        },
        'NStepSarsa_4': {
            'friendly_name': r'semi-gradient $4$-step sarsa',
            'color': next(oranges_palette)
        },
        'NStepSarsa_8': {
            'friendly_name': r'semi-gradient $8$-step sarsa',
            'color': next(oranges_palette)
        },
        'AlwaysAuthenticateAgent': {
            'friendly_name': 'always authenticate',
            'color': next(greens_palette)
        },
        'NeverAuthenticateAgent': {
            'friendly_name': 'never authenticate',
            'color': next(greens_palette)
        },
        'RandomAgent': {
            'friendly_name': 'random',
            'color': next(greens_palette)
        },
    }

    config_dir = os.path.join(parsed_args.results_dir, "config{0:05d}_dir".format(parsed_args.config))
    seed_dirs = [os.path.join(config_dir, seed_dir)
                 for seed_dir in os.listdir(config_dir) if seed_dir.startswith('seed_')]

    fig, ax = plt.subplots(figsize=(11, 3.3))
    fig_percentages, (ax_genuine, ax_fraud) = plt.subplots(ncols=2, nrows=1, figsize=(11, 3.3))

    results_per_alg = {}

    for algorithm, alg_idx in zip(parsed_args.algorithms, range(len(parsed_args.algorithms))):
        results_per_alg[algorithm] = {
            'all_reward_lists': [],
            'all_secondary_auth_genuine_lists': [],
            'all_secondary_auth_fraud_lists': [],
            'all_genuine_lists': [],
            'all_fraud_lists': [],
        }
        rl_agent_dirs = [os.path.join(seed_dir, algorithm) for seed_dir in seed_dirs]

        run_dirs = []

        for rl_agent_dir in rl_agent_dirs:
            if os.path.isdir(rl_agent_dir):
                run_dirs.extend([os.path.join(rl_agent_dir, run_dir)
                                 for run_dir in os.listdir(rl_agent_dir) if run_dir.startswith('run')])

        timesteps = []
        sum_results = []
        index_occurrences = []
        sum_num_auths_genuine = []
        sum_num_auths_fraud = []
        sum_num_genuines = []
        sum_num_frauds = []

        for run_dir in run_dirs:
            with open(os.path.join(run_dir, "cumulative_rewards.csv")) as results_file:
                lines = results_file.readlines()
                results_per_alg[algorithm]['all_reward_lists'].append([])

                if len(lines) == 0:
                    print("0 lines in file: {}".format(os.path.join(run_dir, "cumulative_rewards.csv")))

                if len(lines) < len(timesteps):
                    # don't have as many results in this file as in some other, so need to discard some at the end
                    # should not (and does not :D) happen
                    print("WARNING: cutting off {} lines!".format(len(timesteps) - len(lines)))
                    timesteps = timesteps[:len(lines)]
                    sum_results = sum_results[:len(lines)]
                    index_occurrences = index_occurrences[:len(lines)]

                if 0 < len(timesteps) < len(lines):
                    # have more results in this new file than in some other file, so discard some of these new results
                    # should not (and does not :D) happen
                    print("WARNING: cutting off {} lines!".format(len(lines) - len(timesteps)))
                    lines = lines[:len(timesteps)]

                for line in lines:
                    t, result = line.rstrip('\n').split(", ")
                    t = int(t)
                    result = float(result)
                    results_per_alg[algorithm]['all_reward_lists'][-1].append(result)

                    if t == len(timesteps):
                        timesteps.append(t)
                        sum_results.append(result)
                        index_occurrences.append(1)
                    else:
                        sum_results[t] += result
                        index_occurrences[t] += 1

            with open(os.path.join(run_dir, "num_secondary_auths_genuine.csv")) as secondary_auths_genuine_file:
                lines = secondary_auths_genuine_file.readlines()
                results_per_alg[algorithm]['all_secondary_auth_genuine_lists'].append([])

                for line in lines:
                    t, num = line.rstrip('\n').split(", ")
                    t = int(t)
                    num = int(num)
                    results_per_alg[algorithm]['all_secondary_auth_genuine_lists'][-1].append(num)

                    if t == len(sum_num_auths_genuine):
                        sum_num_auths_genuine.append(num)
                    else:
                        sum_num_auths_genuine[t] += num

            with open(os.path.join(run_dir, "num_secondary_auths_fraudulent.csv")) as secondary_auths_fraud_file:
                lines = secondary_auths_fraud_file.readlines()
                results_per_alg[algorithm]['all_secondary_auth_fraud_lists'].append([])

                for line in lines:
                    t, num = line.rstrip('\n').split(", ")
                    t = int(t)
                    num = int(num)
                    results_per_alg[algorithm]['all_secondary_auth_fraud_lists'][-1].append(num)

                    if t == len(sum_num_auths_fraud):
                        sum_num_auths_fraud.append(num)
                    else:
                        sum_num_auths_fraud[t] += num

            with open(os.path.join(run_dir, "num_genuines.csv")) as genuines_file:
                lines = genuines_file.readlines()
                results_per_alg[algorithm]['all_genuine_lists'].append([])

                for line in lines:
                    t, num = line.rstrip('\n').split(", ")
                    t = int(t)
                    num = int(num)
                    results_per_alg[algorithm]['all_genuine_lists'][-1].append(num)

                    if t == len(sum_num_genuines):
                        sum_num_genuines.append(num)
                    else:
                        sum_num_genuines[t] += num

            with open(os.path.join(run_dir, "num_frauds.csv")) as frauds_file:
                lines = frauds_file.readlines()
                results_per_alg[algorithm]['all_fraud_lists'].append([])

                for line in lines:
                    t, num = line.rstrip('\n').split(", ")
                    t = int(t)
                    num = int(num)
                    results_per_alg[algorithm]['all_fraud_lists'][-1].append(num)

                    if t == len(sum_num_frauds):
                        sum_num_frauds.append(num)
                    else:
                        sum_num_frauds[t] += num

        print("Mean results over {} runs for {}".format(len(run_dirs), algorithm))
        mean_results = [s / len(run_dirs) for s in sum_results]
        results_per_alg[algorithm]['timesteps'] = timesteps
        results_per_alg[algorithm]['sum_results'] = sum_results
        results_per_alg[algorithm]['mean_results'] = mean_results
        results_per_alg[algorithm]['index_occurrences'] = index_occurrences
        results_per_alg[algorithm]['sum_num_auths_genuine'] = sum_num_auths_genuine
        results_per_alg[algorithm]['sum_num_auths_fraud'] = sum_num_auths_fraud
        results_per_alg[algorithm]['sum_num_genuines'] = sum_num_genuines
        results_per_alg[algorithm]['sum_num_frauds'] = sum_num_frauds

        ax.plot(timesteps, mean_results,
                label=algorithms_plot_config[algorithm]['friendly_name'],
                color=algorithms_plot_config[algorithm]['color'])

        mean_num_auths_genuine = np.asarray([s / len(run_dirs) for s in sum_num_auths_genuine])
        mean_num_auths_fraud = np.asarray([s / len(run_dirs) for s in sum_num_auths_fraud])
        mean_num_genuines = np.asarray([s / len(run_dirs) for s in sum_num_genuines])
        mean_num_frauds = np.asarray([s / len(run_dirs) for s in sum_num_frauds])

        # avoid divisions by 0
        mean_num_genuines[mean_num_genuines == 0] = 1
        mean_num_frauds[mean_num_frauds == 0] = 1

        genuine_percentage = 100.0 * (mean_num_auths_genuine / mean_num_genuines)
        results_per_alg[algorithm]['genuine_percentage'] = genuine_percentage

        fraud_percentage = 100.0 * (mean_num_auths_fraud / mean_num_frauds)
        results_per_alg[algorithm]['fraud_percentage'] = fraud_percentage

        ax_genuine.plot(timesteps, genuine_percentage,
                        label=algorithms_plot_config[algorithm]['friendly_name'],
                        color=algorithms_plot_config[algorithm]['color'])

        ax_fraud.plot(timesteps, fraud_percentage,
                      label=algorithms_plot_config[algorithm]['friendly_name'],
                      color=algorithms_plot_config[algorithm]['color'])

        ax_genuine.grid(color='k', linestyle='dotted')
        ax_fraud.grid(color='k', linestyle='dotted')
        ax_fraud.legend(loc=2, fontsize=15, frameon=True).draggable()

        if parsed_args.plot_conf_intervals and \
                (True or algorithm == 'ConcurrentTrueOnlineSarsaLambda' or algorithm == 'NStepSarsa_2'):
            conf_interval_lower = np.zeros(len(mean_results))
            conf_interval_upper = np.zeros(len(mean_results))

            conf_interval_lower_genuine = np.zeros(len(mean_results))
            conf_interval_lower_fraud = np.zeros(len(mean_results))
            conf_interval_upper_genuine = np.zeros(len(mean_results))
            conf_interval_upper_fraud = np.zeros(len(mean_results))

            for i in range(len(mean_results)):
                results_for_timestep = [
                    rewards_list[i] for rewards_list in results_per_alg[algorithm]['all_reward_lists']
                ]

                conf_interval = st.t.interval(0.95, len(results_for_timestep) - 1,
                                              loc=np.mean(results_for_timestep), scale=st.sem(results_for_timestep))

                conf_interval_lower[i] = conf_interval[0]
                conf_interval_upper[i] = conf_interval[1]

                # -------------------------------------------------------------------------------------------------

                genuine_auth_percentages_for_timestep = []
                all_secondary_auth_genuine_lists = results_per_alg[algorithm]['all_secondary_auth_genuine_lists']
                all_genuine_lists = results_per_alg[algorithm]['all_genuine_lists']

                for j in range(len(all_secondary_auth_genuine_lists)):
                    if all_genuine_lists[j][i] > 0:
                        genuine_auth_percentages_for_timestep.append(
                            100.0 * (all_secondary_auth_genuine_lists[j][i] / all_genuine_lists[j][i])
                        )
                    else:
                        genuine_auth_percentages_for_timestep.append(0)

                conf_interval_genuines = st.t.interval(0.95, len(genuine_auth_percentages_for_timestep) - 1,
                                                       loc=np.mean(genuine_auth_percentages_for_timestep),
                                                       scale=st.sem(genuine_auth_percentages_for_timestep))

                conf_interval_lower_genuine[i] = conf_interval_genuines[0]
                conf_interval_upper_genuine[i] = conf_interval_genuines[1]

                # -------------------------------------------------------------------------------------------------

                fraud_auth_percentages_for_timestep = []
                all_secondary_auth_fraud_lists = results_per_alg[algorithm]['all_secondary_auth_fraud_lists']
                all_fraud_lists = results_per_alg[algorithm]['all_fraud_lists']

                for j in range(len(all_secondary_auth_fraud_lists)):
                    if all_fraud_lists[j][i] > 0:
                        fraud_auth_percentages_for_timestep.append(
                            100.0 * (all_secondary_auth_fraud_lists[j][i] / all_fraud_lists[j][i])
                        )
                    else:
                        fraud_auth_percentages_for_timestep.append(0)

                conf_interval_frauds = st.t.interval(0.95, len(fraud_auth_percentages_for_timestep) - 1,
                                                     loc=np.mean(fraud_auth_percentages_for_timestep),
                                                     scale=st.sem(fraud_auth_percentages_for_timestep))

                conf_interval_lower_fraud[i] = conf_interval_frauds[0]
                conf_interval_upper_fraud[i] = conf_interval_frauds[1]

            ax.fill_between(timesteps,
                            conf_interval_lower,
                            conf_interval_upper,
                            alpha=0.25, edgecolor=algorithms_plot_config[algorithm]['color'],
                            facecolor=algorithms_plot_config[algorithm]['color'])

            ax_genuine.fill_between(timesteps,
                                    conf_interval_lower_genuine,
                                    conf_interval_upper_genuine,
                                    alpha=0.25, edgecolor=algorithms_plot_config[algorithm]['color'],
                                    facecolor=algorithms_plot_config[algorithm]['color'])

            ax_fraud.fill_between(timesteps,
                                  conf_interval_lower_fraud,
                                  conf_interval_upper_fraud,
                                  alpha=0.25, edgecolor=algorithms_plot_config[algorithm]['color'],
                                  facecolor=algorithms_plot_config[algorithm]['color'])

    '''
    import statsmodels.stats.api as sms
    rewards_lists_1 = results_per_alg['ConcurrentTrueOnlineSarsaLambda']['all_reward_lists']
    rewards_lists_2 = results_per_alg['NStepSarsa_2']['all_reward_lists']
    x1 = [rewards_list[-1] for rewards_list in rewards_lists_1]
    x2 = [rewards_list[-1] for rewards_list in rewards_lists_2]

    cm = sms.CompareMeans(sms.DescrStatsW(x1), sms.DescrStatsW(x2))
    print(cm.tconfint_diff(usevar='unequal'))
    '''

    ax.grid(color='k', linestyle='dotted')
    ax.legend(loc=2, fontsize=15, frameon=True).draggable()

    if parsed_args.out_dir is None:
        plt.show()
