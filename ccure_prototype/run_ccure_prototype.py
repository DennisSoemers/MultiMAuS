"""
This script runs a full prototype for the C-Cure project

Info on the C-Cure project: http://www.securit-brussels.be/project/c-cure/

Main author for this script:
    @author Dennis Soemers (VUB AI Lab)

Many other people also contributed to the project from
Vrije Universiteit Brussel (AI Lab and BUTO) and Université Catholique de Louvain (Crypto Group)
"""

import argparse
import random
import json
import logging
import multiprocessing
import numpy as np
import os
import threading
import time
from datetime import datetime
from mesa.time import BaseScheduler

from authenticators.simple_authenticators import NeverSecondAuthenticator
from ccure_prototype.experiment_summary import ExperimentSummary
from ccure_prototype.gui.control_frame import create_control_frame, is_control_frame_alive
from ccure_prototype.rl_authenticator import RLAuthenticator
from ccure_prototype.rl_authenticator import StateCreator
from ccure_prototype.rl.always_authenticate_agent import AlwaysAuthenticateAgent
from ccure_prototype.rl.concurrent_true_online_sarsa_lambda_agent import ConcurrentTrueOnlineSarsaLambdaAgent
from ccure_prototype.rl.n_step_sarsa_agent import NStepSarsaAgent
from ccure_prototype.rl.never_authenticate_agent import NeverAuthenticateAgent
from ccure_prototype.rl.oracle_agent import OracleAgent
from ccure_prototype.rl.random_agent import RandomAgent
from ccure_prototype.rl.sarsa_agent import SarsaAgent
from data.features.aggregate_features import AggregateFeatures
from data.features.apate_graph_features import ApateGraphFeatures
from simulator import parameters
from simulator.transaction_model import TransactionModel


if __name__ == '__main__':
    # want stack trace when warnings are raised
    np.seterr(all='raise')

    logger = multiprocessing.log_to_stderr(logging.INFO)

    parser = argparse.ArgumentParser(description='C-Cure Prototype')
    parser.add_argument('--num-sim-steps-training-data', dest='num_sim_steps_training_data', type=int, default=6000,
                        help='Number of simulation steps to run when generating training data.')
    parser.add_argument('--train-feature-engineering-ratio', dest='train_feature_engineering_ratio', type=float,
                        default=0.125,
                        help='Ratio of training data that should be used to train feature engineering (remainder is'
                             ' used to train model).')
    parser.add_argument('--num-sim-steps-skip', dest='num_sim_steps_skip', type=int, default=1000,
                        help='Number of simulation steps to run to generate skip/gap data (in between training'
                             ' and test data).')
    parser.add_argument('--num-sim-steps-evaluation', dest='num_sim_steps_evaluation', type=int, default=10_000,
                        help='Number of simulation steps to run to generate data for evaluation (test data).')

    parser.add_argument('--flat-fee', dest='flat_fee', type=float, default=0.01,
                        help='Flat fee we take for every successful transaction.')
    parser.add_argument('--relative-fee', dest='relative_fee', type=float, default=0.01,
                        help='Relative fee we take for every successful transaction'
                             ' (multiplied by transaction amount).')

    parser.add_argument('--use-seed-specific-models', dest='use_seed_specific_models', action='store_true',
                        help='Train separate offline models for every random seed (otherwise, we share trained'
                             ' models across different runs with different random seeds).')

    parser.add_argument('--rl-agent', dest='rl_agent',
                        choices=[
                            'Sarsa',
                            'ConcurrentTrueOnlineSarsaLambda',
                            'ConcurrentTrueOnlineSarsaLambda_06',
                            'ConcurrentTrueOnlineSarsaLambda_07',
                            'ConcurrentTrueOnlineSarsaLambda_09',
                            'AlwaysAuthenticateAgent',
                            'NeverAuthenticateAgent',
                            'OracleAgent',
                            'NStepSarsa_1',
                            'NStepSarsa_2',
                            'NStepSarsa_4',
                            'NStepSarsa_8',
                            'RandomAgent'
                        ],
                        default='NStepSarsa_4',
                        help='RL Agent to use during evaluation phase.')

    parser.add_argument('--cs-models-r-filepath', dest='cs_models_r_filepath',
                        default='D:/Apps/C-Cure/Code/R/optimized_CSModels.R',
                        help='Filepath to R script for Cost-Sensitive models.')

    parser.add_argument('--seed', dest='seed', type=int, default=None,
                        help='Seed for RNG. By default, will randomly generate one through RNG based on default seed.')

    parser.add_argument('--run-idx', dest='run_idx', type=int, default=None,
                        help='Unique index for run. By default, will try to automatically find the next index'
                             ' based on which directories for results already exist. The default behaviour may go'
                             ' wrong in cases where we start many different runs in parallel.')

    parser.add_argument('--out-dir', dest='out_dir', default=os.path.join(os.path.dirname(__file__), 'results_test'),
                        help='Base directory for results output.')

    parser.add_argument('--disable-ui', dest='disable_ui', action='store_true',
                        help='Disable the UI (control panel).')

    parsed_args = parser.parse_args()

    '''
    --------------------------------------------------------------------------------------
    Configuration
    --------------------------------------------------------------------------------------
    '''

    # -------------------------------------------
    # algorithms / models / components to use
    # -------------------------------------------
    # authenticator to use during the phase where we generate training / skip data
    authenticator_training_phase = NeverSecondAuthenticator()
    authenticator_test_phase = None

    # -------------------------------------------
    # experiment setup
    # -------------------------------------------
    # number of steps to simulate to generate training data
    NUM_SIM_STEPS_TRAINING_DATA = parsed_args.num_sim_steps_training_data
    # ratio out of training data to use for feature engineering (necessary to learn how to create APATE graph features)
    TRAIN_FEATURE_ENGINEERING_RATIO = parsed_args.train_feature_engineering_ratio

    # number of steps to simulate and discard afterwards to create gap between training and test data
    NUM_SIM_STEPS_SKIP = parsed_args.num_sim_steps_skip

    # number of steps to simulate for evaluation
    NUM_SIM_STEPS_EVALUATION = parsed_args.num_sim_steps_evaluation

    # flat fee we take for every genuine transaction
    FLAT_FEE = parsed_args.flat_fee
    # relative fee (multiplied by transaction amount) we take for every genuine transaction
    RELATIVE_FEE = parsed_args.relative_fee

    # if True, we'll share trained R models among all seeds with otherwise the same config
    USE_SEED_AGNOSTIC_MODELS = not parsed_args.use_seed_specific_models

    RL_AGENT = parsed_args.rl_agent

    # if True, we'll also profile our running code
    PROFILE = False

    # -------------------------------------------
    # R offline models
    # -------------------------------------------
    # filepath for offline models R script.
    CS_MODELS_R_FILEPATH = parsed_args.cs_models_r_filepath

    # -------------------------------------------
    # simulator parameters
    # -------------------------------------------
    simulator_params = parameters.get_default_parameters()
    simulator_params['end_date'] = datetime(9999, 12, 31)
    simulator_params['stay_prob'][0] = 0.9      # stay probability for genuine customers
    simulator_params['stay_prob'][1] = 0.96     # stay probability for fraudsters

    if parsed_args.seed is None:
        simulator_params['seed'] = random.randrange(2**31)  # only 2^31 instead of 2^32 because R cant handle big seeds
    else:
        simulator_params['seed'] = parsed_args.seed

    seed = simulator_params['seed']
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ": Running C-Cure prototype with seed = ", seed)

    import rpy2
    print("rpy2 version: {}".format(rpy2.__version__))
    from rpy2.rinterface import R_VERSION_BUILD
    print("R_VERSION_BUILD: {}".format(R_VERSION_BUILD))

    simulator_params['num_fraudsters'] = 300
    simulator_params['stay_prob'][0] = 0.99     # genuine
    simulator_params['stay_prob'][1] = 0.96     # fraud

    simulator_params['flat_fee'] = FLAT_FEE
    simulator_params['relative_fee'] = RELATIVE_FEE

    # we assume fraudulent cases get reported after this many simulator steps (hours)   TODO can make this much more fancy
    FRAUD_REPORT_TIME = 72

    # -------------------------------------------
    # output
    # -------------------------------------------
    OUTPUT_DIR = parsed_args.out_dir

    experiment_config = simulator_params.copy()
    experiment_config.pop('seed')
    config_idx = 0
    while True:
        if os.path.isdir(os.path.join(OUTPUT_DIR, "config{0:05d}_dir".format(config_idx))):
            existing_config = json.load(open(os.path.join(OUTPUT_DIR, "config{0:05d}_file.json".format(config_idx))))

            # we do the loads(dumps( . )) here because datetime objects will just be strings in existing_config
            if json.loads(json.dumps(experiment_config, default=str)) == existing_config:
                # matching config, so use the existing dictionary
                break
            else:
                # this particular existing config doesn't match, so increment index
                config_idx += 1
        else:
            # haven't found a matching config, so need to save new file and create new dir
            with open(os.path.join(OUTPUT_DIR, "config{0:05d}_file.json".format(config_idx)), 'w') as outfile:
                json.dump(experiment_config, outfile, indent=4, default=str)

            break

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "config{0:05d}_dir".format(config_idx))

    if USE_SEED_AGNOSTIC_MODELS:
        OUTPUT_DIR_SHARED_RUNS = OUTPUT_DIR

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'seed_{}'.format(simulator_params['seed']))

    if not USE_SEED_AGNOSTIC_MODELS:
        OUTPUT_DIR_SHARED_RUNS = OUTPUT_DIR

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, RL_AGENT)

    if parsed_args.run_idx is None:
        results_run_idx = 0
    else:
        results_run_idx = parsed_args.run_idx

    while True:
        if os.path.isdir(os.path.join(OUTPUT_DIR, "run{0:05d}".format(results_run_idx))):
            results_run_idx += 1
        else:
            break

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "run{0:05d}".format(results_run_idx))
    os.makedirs(OUTPUT_DIR)
    print("Output dir = {}".format(OUTPUT_DIR))

    OUTPUT_FILE_FEATURE_LEARNING_DATA = os.path.join(OUTPUT_DIR, 'feature_learning_data.csv')
    OUTPUT_FILE_MODEL_LEARNING_DATA = os.path.join(OUTPUT_DIR, 'model_learning_data.csv')

    # -------------------------------------------
    # User Interface
    # -------------------------------------------
    # after how many simulation steps should we update our control UI?
    CONTROL_UI_UPDATE_FREQUENCY = 10

    # after how many simulation steps should we update our estimate of simulation speed?
    UPDATE_SPEED_FREQUENCY = 100

    # after how many simulation steps should we update our control UI during evaluation?
    CONTROL_UI_UPDATE_FREQUENCY_EVAL = 1

    # after how many simulation steps should we update our estimate of simulation speed during evaluation?
    UPDATE_SPEED_FREQUENCY_EVAL = 1

    # number of models we expect to train in R. This param is only used to report progress in GUI
    NUM_R_MODELS = 62

    '''
    --------------------------------------------------------------------------------------
    Global Variables
    --------------------------------------------------------------------------------------
    '''
    # we will later create our simulation model in this variable
    simulator = None
    # we will later create an object that can construct APATE graph features in this variable
    apate_graph_feature_constructor = None
    # we will later create an object that can construct aggregate features in this variable
    aggregate_feature_constructor = None


    '''
    --------------------------------------------------------------------------------------
    Helper functions
    --------------------------------------------------------------------------------------
    '''

    # -------------------------------------------
    # handling data generated by simulator
    # -------------------------------------------
    def block_cards(card_ids, replace_fraudsters=True):
        """
        Blocks the given list of Card IDs (removing all genuine and fraudulent customers with matching
        Card IDs from the simulation).

        NOTE: This function is only intended to be called using Card IDs that are 100% known to have
        been involved in fraudulent transactions. If the list contains more than a single Card ID,
        and the Card ID has not been used in any fraudulent transactions, the function may not be able
        to find the matching customer (due to an optimization in the implementation)

        :param card_ids:
            List of one or more Card IDs to block
        :param replace_fraudsters:
            If True, also replaces the banned fraudsters by an equal number of new fraudsters. True by default
        """
        n = len(card_ids)

        if n == 0:
            # nothing to do
            return

        num_banned_fraudsters = 0

        if n == 1:
            # most efficient implementation in this case is simply to loop once through all customers (fraudulent
            # as well as genuine) and compare to our single blocked card ID
            blocked_card_id = card_ids[0]

            for customer in simulator.customers:
                if customer.card_id == blocked_card_id:
                    customer.stay = False

                    # should not be any more customers with same card ID, so can break
                    break

            for fraudster in simulator.fraudsters:
                if fraudster.card_id == blocked_card_id:
                    fraudster.stay = False
                    num_banned_fraudsters += 1

                    # should not be any more fraudsters with same card ID, so can break
                    break
        else:
            # with naive implementation, we'd have n loops through the entire list of customers, which may be expensive
            # instead, we loop through it once to collect only those customers with corrupted cards. Then, we follow
            # up with n loops through that much smaller list of customers with corrupted cards
            compromised_customers = [c for c in simulator.customers if c.card_corrupted]

            for blocked_card_id in card_ids:
                for customer in compromised_customers:
                    if customer.card_id == blocked_card_id:
                        customer.stay = False

                        # should not be any more customers with same card ID, so can break
                        break

                for fraudster in simulator.fraudsters:
                    if fraudster.card_id == blocked_card_id:
                        fraudster.stay = False
                        num_banned_fraudsters += 1

                        # should not be any more fraudsters with same card ID, so can break
                        break

        if replace_fraudsters:
            simulator.add_fraudsters(num_banned_fraudsters)


    def clear_log():
        """
        Clears all transactions generated so far from memory
        """
        agent_vars = simulator.log_collector.agent_vars
        for reporter_name in agent_vars:
            agent_vars[reporter_name] = []


    def get_log(clear_after=True):
        """
        Returns a log (in the form of a pandas dataframe) of the transactions generated so far.

        :param clear_after:
            If True, will clear the transactions from memory. This means that subsequent calls to get_log()
            will no longer include the transactions that have already been returned in a previous call.
        :return:
            The logged transactions. Returns None if no transactions were logged
        """
        log = simulator.log_collector.get_agent_vars_dataframe()

        if log is None:
            return None

        log.reset_index(drop=True, inplace=True)

        if clear_after:
            clear_log()

        return log


    def on_customer_leave(card_id, customer):
        if authenticator_test_phase is not None:
            authenticator_test_phase.agent.on_customer_leave(card_id, customer)


    # -------------------------------------------
    # feature engineering
    # -------------------------------------------
    def process_data(data):
        """
        Processes the given data, so that it will be ready for use in Machine Learning models. New features
        are added by the feature constructors, features which are no longer necessary afterwards are removed,
        and the Target feature is moved to the back of the dataframe

        NOTE: processing is done in-place

        :param data:
            Data to process
        :return:
            Processed dataframe
        """
        apate_graph_feature_constructor.add_graph_features(data)
        data = aggregate_feature_constructor.add_aggregate_features(data)

        # remove non-numeric columns / columns we don't need after adding features
        data.drop(["Global_Date", "Local_Date", "MerchantID", "Currency", "Country",
                   "AuthSteps", "TransactionCancelled", "TransactionSuccessful"],
                  inplace=True, axis=1)

        # move CardID, TimeSinceLastTransaction, and Target columns to the end
        data = data[
            [col for col in data if col != "Target" and col != "CardID" and col != "TimeSinceLastTransaction"] +
            ["CardID", "TimeSinceLastTransaction", "Target"]
        ]

        return data


    def update_feature_constructors_unlabeled(data):
        """
        Performs an update of existing feature constructors, treating the given new data
        as being unlabeled.

        :param data:
            (unlabeled) new data (should NOT have been passed into prepare_feature_constructors() previously)
        """
        global aggregate_feature_constructor    # need to make this explicitly global for Feature Eng. Thread

        if data is not None:
            aggregate_feature_constructor.update_unlabeled(data)


    '''
    --------------------------------------------------------------------------------------
    The script's main code
    --------------------------------------------------------------------------------------
    '''
    # profiling
    if PROFILE:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    # construct our simulation model
    simulator = TransactionModel(simulator_params,
                                 authenticator=authenticator_training_phase,
                                 scheduler=BaseScheduler(None))   # NOTE: not using random agent schedule

    simulator.customer_leave_callbacks.append(on_customer_leave)

    # GUI
    if not parsed_args.disable_ui:
        control_frame = create_control_frame()
        timestep_speed_measure_time_start = time.time()
        start_time_generate_training_data = timestep_speed_measure_time_start
        timestep_speed = 0

    exit_simulation = False

    # generate training data
    print("Generating training data...")
    for t in range(NUM_SIM_STEPS_TRAINING_DATA):
        if not parsed_args.disable_ui:
            if t % CONTROL_UI_UPDATE_FREQUENCY == 0:
                if not is_control_frame_alive():
                    exit_simulation = True
                    break

                control_frame.update_info_generate_training_data(t, NUM_SIM_STEPS_TRAINING_DATA, timestep_speed)
                control_frame.root.update()

                if control_frame.want_quit:
                    exit_simulation = True
                    break
                if control_frame.want_pause:
                    time.sleep(1)
                    continue

        # This line is the only code we actually want to execute in the loop, the simulator step
        simulator.step()

        if not parsed_args.disable_ui:
            if t % UPDATE_SPEED_FREQUENCY == 0:
                curr_time = time.time()
                timestep_speed = UPDATE_SPEED_FREQUENCY / (curr_time - timestep_speed_measure_time_start)
                timestep_speed_measure_time_start = curr_time

    if not exit_simulation:
        training_data = get_log(clear_after=True)
        num_training_instances = training_data.shape[0]
        num_feature_learning_instances = int(num_training_instances * TRAIN_FEATURE_ENGINEERING_RATIO)
        num_model_learning_instances = num_training_instances - num_feature_learning_instances
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ": Generated ",
              num_training_instances, " training instances (",
              num_feature_learning_instances, " for feature learning, ",
              num_model_learning_instances, " for model learning)")

        if not parsed_args.disable_ui:
            control_frame.update_info_generate_training_data(NUM_SIM_STEPS_TRAINING_DATA, NUM_SIM_STEPS_TRAINING_DATA,
                                                             timestep_speed,
                                                             total_time=time.time() - start_time_generate_training_data)
            control_frame.root.update()

        feature_learning_data = training_data.iloc[:num_feature_learning_instances]
        model_learning_data = training_data.iloc[num_feature_learning_instances:]

        need_train_models = not os.path.isdir(os.path.join(OUTPUT_DIR_SHARED_RUNS, "Models"))

        # feature engineering
        print("Feature engineering...")

        # define feature engineering function that we can run in a different thread without blocking this one
        def feature_engineering_func(feature_learning_data, model_learning_data, out_list, need_train_models):
            global aggregate_feature_constructor
            global apate_graph_feature_constructor
            aggregate_feature_constructor = AggregateFeatures(feature_learning_data)
            apate_graph_feature_constructor = ApateGraphFeatures(feature_learning_data)
            update_feature_constructors_unlabeled(model_learning_data)

            # compute features for our model learning data
            if need_train_models:
                model_learning_data = process_data(model_learning_data)

            # store in out_list so we can retrieve results in main thread
            out_list[0] = aggregate_feature_constructor
            out_list[1] = apate_graph_feature_constructor
            out_list[2] = model_learning_data

        # create new thread
        out_list = [None] * 3
        feature_eng_thread = threading.Thread(target=feature_engineering_func, name="Feature Eng. Thread",
                                              args=(feature_learning_data,
                                                    model_learning_data,
                                                    out_list,
                                                    need_train_models),
                                              daemon=True)
        start_time_feature_engineering = time.time()
        feature_eng_thread.start()

        while feature_eng_thread.is_alive():
            if not parsed_args.disable_ui:
                if not is_control_frame_alive():
                    exit_simulation = True
                    break

                control_frame.update_info_feature_engineering(time.time() - start_time_feature_engineering, finished=False)
                control_frame.root.update()

                if control_frame.want_quit:
                    exit_simulation = True
                    break

            # sleep for a bit while we let feature eng. thread do its work
            time.sleep(1)

        if not exit_simulation:
            aggregate_feature_constructor = out_list[0]
            apate_graph_feature_constructor = out_list[1]
            model_learning_data = out_list[2]

            feature_learning_data.to_csv(OUTPUT_FILE_FEATURE_LEARNING_DATA, index_label=False)
            model_learning_data.to_csv(OUTPUT_FILE_MODEL_LEARNING_DATA, index_label=False)

            feature_eng_thread = None   # this may help clean up?

            if not parsed_args.disable_ui:
                control_frame.update_info_feature_engineering(time.time() - start_time_feature_engineering, finished=True)
                control_frame.root.update()

            # train offline models in R if our Models directory does not already exist
            start_time_r_model_training = time.time()
            if need_train_models:
                print("Training R models...")

                # create new process
                from ccure_prototype import r_model_training
                train_r_process = multiprocessing.Process(target=r_model_training.train_r_models,
                                                          name="R Model Training Process",
                                                          args=(CS_MODELS_R_FILEPATH,
                                                                OUTPUT_FILE_MODEL_LEARNING_DATA,
                                                                OUTPUT_DIR_SHARED_RUNS,
                                                                seed))
                train_r_process.daemon = True
                train_r_process.start()

                while train_r_process.is_alive():
                    if not parsed_args.disable_ui:
                        if not is_control_frame_alive():
                            exit_simulation = True
                            break

                    # figure out for how many models we already have files
                    trial_models_dir = os.path.join(OUTPUT_DIR_SHARED_RUNS, "TrialModels")
                    final_models_dir = os.path.join(OUTPUT_DIR_SHARED_RUNS, "Models")

                    if not os.path.isdir(trial_models_dir) or not os.path.isdir(final_models_dir):
                        num_models_trained = 0
                    else:
                        num_models_trained = len(os.listdir(trial_models_dir)) + len(os.listdir(final_models_dir))

                    if not parsed_args.disable_ui:
                        control_frame.update_info_r_model_training(
                            num_models_trained,
                            NUM_R_MODELS, time.time() - start_time_r_model_training)
                        control_frame.root.update()

                        if control_frame.want_quit:
                            exit_simulation = True
                            break

                    # sleep for a bit while we let model training thread do its work
                    time.sleep(1)

            if not exit_simulation:
                print("Using R models from directory: {}".format(OUTPUT_DIR_SHARED_RUNS))

                trial_models_dir = os.path.join(OUTPUT_DIR_SHARED_RUNS, "TrialModels")
                final_models_dir = os.path.join(OUTPUT_DIR_SHARED_RUNS, "Models")

                if not parsed_args.disable_ui:
                    control_frame.update_info_r_model_training(
                        len(os.listdir(trial_models_dir)) + len(os.listdir(final_models_dir)),
                        NUM_R_MODELS, time.time() - start_time_r_model_training)
                    control_frame.root.update()

                # load R models into memory
                print("importing rpy2.rinterface...")
                import rpy2.rinterface as ri
                print("calling initr()...")
                ri.initr()
                r_set_seed = ri.baseenv['set.seed']
                print("calling R set_seed()")
                r_set_seed(seed)
                print("parsing R commands...")
                ri.parse('library(compiler)')
                ri.parse('enableJIT(3)')
                r_source = ri.baseenv['source']
                print("calling source({})...".format(CS_MODELS_R_FILEPATH))
                r_source(ri.StrSexpVector((CS_MODELS_R_FILEPATH, )))
                savepath_string = OUTPUT_DIR_SHARED_RUNS.replace("\\", "/")
                if not savepath_string.endswith("/"):
                    savepath_string += "/"
                r_loadCSModels = ri.globalenv['loadCSModels']
                print("loading CS models...")
                num_r_predictions = int(r_loadCSModels(ri.StrSexpVector((savepath_string, )))[0])
                r_getModelConfigNames = ri.globalenv['getModelConfigNames']
                cs_model_config_names = r_getModelConfigNames()
                cs_model_config_names = [cs_model_config_names[idx] for idx in range(num_r_predictions)]
                r_predictCSModels = ri.globalenv['predictCSModels']
                r_predictCSModelsSlow = ri.globalenv['predictCSModelsSlow']

                # create function that we can use to make predictions for transactions
                def make_predictions(feature_vector):
                    #print("feature_vector = ", feature_vector)
                    preds = np.asarray(r_predictCSModels(ri.FloatSexpVector(feature_vector)))

                    '''
                    if random.random() < 0.01:
                        # in about 1% of the predictions, also run a slow prediction and make sure it's equal to the
                        # optimized version
                        # TODO remove this
                        slow_preds = np.asarray(r_predictCSModelsSlow(ri.FloatSexpVector(feature_vector)))

                        if not np.array_equal(preds, slow_preds):
                            print("WARNING: slow R predictions not equal to optimized R predictions!")
                            print("Slow predictions: ", slow_preds)
                            print("Fast predictions: ", preds)
                            print("Feature vector: ", feature_vector)
                            print("")
                    '''

                    preds[np.isnan(preds)] = 0

                    return preds

                # get rid of all fraudsters in training data and replace them with new fraudsters
                print("replacing known fraudsters...")
                fraudster_ids = set()
                for transaction in training_data.itertuples():  # this loops through feature and model learning data at once
                    if transaction.Target == 1:
                        fraudster_ids.add(transaction.CardID)

                block_cards(list(fraudster_ids), replace_fraudsters=True)
                fraudster_ids = None    # clean memory

                # generate some data as a gap between training and test data
                print("Generating gap data...")
                start_time_generate_gap_data = time.time()

                for t in range(NUM_SIM_STEPS_SKIP):
                    if not parsed_args.disable_ui:
                        if t % CONTROL_UI_UPDATE_FREQUENCY == 0:
                            if not is_control_frame_alive():
                                exit_simulation = True
                                break

                            control_frame.update_info_generate_gap_data(t, NUM_SIM_STEPS_SKIP, timestep_speed)
                            control_frame.root.update()

                            if control_frame.want_quit:
                                exit_simulation = True
                                break
                            if control_frame.want_pause:
                                time.sleep(1)
                                continue

                    # This line is the only code we actually want to execute in the loop, the simulator step
                    simulator.step()

                    if not parsed_args.disable_ui:
                        if t % UPDATE_SPEED_FREQUENCY == 0:
                            curr_time = time.time()
                            timestep_speed = UPDATE_SPEED_FREQUENCY / (curr_time - timestep_speed_measure_time_start)
                            timestep_speed_measure_time_start = curr_time

                if not exit_simulation:
                    skip_data = get_log(clear_after=True)

                    # can still use the skip data to update our feature engineering
                    update_feature_constructors_unlabeled(skip_data)

                    if not parsed_args.disable_ui:
                        control_frame.update_info_generate_gap_data(NUM_SIM_STEPS_SKIP,
                                                                    NUM_SIM_STEPS_SKIP,
                                                                    timestep_speed,
                                                                    total_time=time.time() - start_time_generate_gap_data)
                        control_frame.root.update()

                    # allow all training and skip data to be cleaned from memory
                    # (may not actually matter since AggregateFeatures keeps them all stored anyway)
                    training_data = None
                    feature_learning_data = None
                    model_learning_data = None
                    skip_data = None

                    # stop tracking bounds for scaling RL features
                    simulator.track_max_values = False

                    # create our RL-based authenticator and add it to the simulator
                    state_creator = StateCreator(make_predictions_func=make_predictions,
                                                 feature_processing_func=process_data,
                                                 num_models=num_r_predictions)

                    if RL_AGENT == "Sarsa":
                        rl_agent = SarsaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features()
                        )
                    elif RL_AGENT == "ConcurrentTrueOnlineSarsaLambda":
                        rl_agent = ConcurrentTrueOnlineSarsaLambdaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features()
                        )
                    elif RL_AGENT == "ConcurrentTrueOnlineSarsaLambda_06":
                        rl_agent = ConcurrentTrueOnlineSarsaLambdaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            lambda_=0.6
                        )
                    elif RL_AGENT == "ConcurrentTrueOnlineSarsaLambda_07":
                        rl_agent = ConcurrentTrueOnlineSarsaLambdaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            lambda_=0.7
                        )
                    elif RL_AGENT == "ConcurrentTrueOnlineSarsaLambda_09":
                        rl_agent = ConcurrentTrueOnlineSarsaLambdaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            lambda_=0.9
                        )
                    elif RL_AGENT == "AlwaysAuthenticateAgent":
                        rl_agent = AlwaysAuthenticateAgent(
                            num_actions=3,
                            num_state_features=state_creator.get_num_state_features()
                        )
                    elif RL_AGENT == "NeverAuthenticateAgent":
                        rl_agent = NeverAuthenticateAgent(
                            num_actions=3,
                            num_state_features=state_creator.get_num_state_features()
                        )
                    elif RL_AGENT == "OracleAgent":
                        rl_agent = OracleAgent(
                            num_actions=3,
                            num_state_features=state_creator.get_num_state_features()
                        )
                    elif RL_AGENT == "NStepSarsa_1":
                        rl_agent = NStepSarsaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            n=1
                        )
                    elif RL_AGENT == "NStepSarsa_2":
                        rl_agent = NStepSarsaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            n=2
                        )
                    elif RL_AGENT == "NStepSarsa_4":
                        rl_agent = NStepSarsaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            n=4
                        )
                    elif RL_AGENT == "NStepSarsa_8":
                        rl_agent = NStepSarsaAgent(
                            num_real_actions=2, num_actions=3,
                            num_state_features=state_creator.get_num_state_features(),
                            n=8
                        )
                    elif RL_AGENT == "RandomAgent":
                        rl_agent = RandomAgent(
                            num_actions=3,
                            num_state_features=state_creator.get_num_state_features()
                        )
                    else:
                        print("UNKNOWN RL AGENT: ", RL_AGENT)
                        rl_agent = None

                    authenticator_test_phase = RLAuthenticator(
                        agent=rl_agent,
                        state_creator=state_creator, flat_fee=FLAT_FEE, relative_fee=RELATIVE_FEE,
                        cs_model_config_names=cs_model_config_names,
                        simulator=simulator
                    )

                    simulator.authenticator = authenticator_test_phase

                    # in this dict, we'll store for simulation steps (key = t) lists of transactions that need to be reported
                    # at that time
                    to_report = {}

                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), " (run {}): Starting evaluation...".format(results_run_idx))
                    timestep_speed_measure_time_start = time.time()

                    with ExperimentSummary(flat_fee=FLAT_FEE,
                                           relative_fee=RELATIVE_FEE,
                                           output_dir=OUTPUT_DIR,
                                           cs_model_config_names=cs_model_config_names,
                                           rl_authenticator=authenticator_test_phase,
                                           rl_agent=rl_agent) as summary:

                        t = 0

                        while t < NUM_SIM_STEPS_EVALUATION:
                            if t % 500 == 0:
                                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), " (run {}): t = {}".format(results_run_idx, t))

                            if not parsed_args.disable_ui:
                                if t % CONTROL_UI_UPDATE_FREQUENCY_EVAL == 0:
                                    if not is_control_frame_alive():
                                        break

                                    control_frame.update_info_labels_eval(
                                        t, NUM_SIM_STEPS_EVALUATION, timestep_speed,
                                        num_transactions=summary.num_transactions[-1],
                                        num_genuines=summary.num_genuines[-1],
                                        num_fraudulents=summary.num_frauds[-1],
                                        num_secondary_auths=summary.num_secondary_auths[-1],
                                        num_secondary_auths_genuine=summary.num_secondary_auths_genuine[-1],
                                        num_secondary_auths_fraud=summary.num_secondary_auths_fraudulent[-1])

                                    control_frame.root.update()

                                    if control_frame.want_quit:
                                        break
                                    if control_frame.want_pause:
                                        time.sleep(1)
                                        continue

                            # process any transactions that get reported and turn out to be fraudulent at this time
                            if t in to_report:
                                transactions_to_report = to_report.pop(t)

                                for transaction in transactions_to_report:
                                    authenticator_test_phase.update_fraudulent(transaction)

                            # a single simulator step
                            simulator.step()

                            # let our summary know we started a new timestep
                            summary.new_timestep(t)

                            # dataframe for the latest step
                            # (can contain multiple transactions generated in same simulator step)
                            new_data = get_log(clear_after=True)

                            old_num_genuines = summary.num_genuines[-1]

                            if new_data is not None:
                                # use it for unlabeled feature updates
                                update_feature_constructors_unlabeled(new_data)

                                for row in new_data.iterrows():
                                    transaction = row[1]
                                    if transaction.Target == 1:
                                        # fraudulent transaction, report it FRAUD_REPORT_TIME sim-steps from now
                                        if t + FRAUD_REPORT_TIME in to_report:
                                            to_report[t + FRAUD_REPORT_TIME].append(transaction)
                                        else:
                                            to_report[t + FRAUD_REPORT_TIME] = [transaction, ]

                                    summary.record_transaction(transaction)

                            # the loop through data above does not include transactions that were filtered out by
                            # secondary authentication, so need to correct for that in our summary
                            num_new_blocked_genuines = \
                                authenticator_test_phase.num_secondary_auths_blocked_genuines \
                                - summary.num_secondary_auths_blocked_genuine[-1]
                            summary.num_transactions[-1] += num_new_blocked_genuines
                            summary.num_genuines[-1] += num_new_blocked_genuines
                            summary.num_secondary_auths_genuine[-1] = \
                                authenticator_test_phase.num_secondary_auths_passed_genuines + \
                                authenticator_test_phase.num_secondary_auths_blocked_genuines
                            summary.num_secondary_auths_blocked_genuine[-1] = \
                                authenticator_test_phase.num_secondary_auths_blocked_genuines

                            num_new_blocked_frauds = \
                                authenticator_test_phase.num_secondary_auths_blocked_frauds - \
                                summary.num_secondary_auths_fraudulent[-1]
                            summary.num_transactions[-1] += num_new_blocked_frauds
                            summary.num_frauds[-1] += num_new_blocked_frauds
                            summary.num_secondary_auths_fraudulent[-1] = \
                                authenticator_test_phase.num_secondary_auths_blocked_frauds

                            summary.num_secondary_auths[-1] = authenticator_test_phase.num_secondary_auths

                            summary.total_population[-1] = len(simulator.customers) + len(simulator.fraudsters)
                            summary.genuine_population[-1] = len(simulator.customers)
                            summary.fraud_population[-1] = len(simulator.fraudsters)

                            summary.total_fraud_amounts_seen[-1] = authenticator_test_phase.total_fraud_amounts_seen

                            curr_rl_weights = rl_agent.get_weights()
                            for action in range(summary.num_actions):
                                for w_idx in range(summary.num_weights_per_action):
                                    summary.weights_per_action[action][w_idx][-1] = \
                                        curr_rl_weights[w_idx + action * summary.num_weights_per_action]

                            for m in cs_model_config_names:
                                model_summ = authenticator_test_phase.cs_model_performance_summaries[m]
                                summary.num_true_positives_per_model[m][-1] = model_summ.num_true_positives
                                summary.num_false_positives_per_model[m][-1] = model_summ.num_false_positives
                                summary.num_true_negatives_per_model[m][-1] = model_summ.num_true_negatives
                                summary.num_false_negatives_per_model[m][-1] = model_summ.num_false_negatives
                                summary.total_fraud_amounts_detected_per_model[m][-1] = \
                                    model_summ.total_fraud_amounts_detected
                                summary.num_agreements_per_model[m][-1] = model_summ.num_agreements
                                summary.num_agreements_true_positive_per_model[m][-1] = \
                                    model_summ.num_agreements_true_positive
                                summary.num_agreements_false_positive_per_model[m][-1] = \
                                    model_summ.num_agreements_false_positive
                                summary.num_agreements_true_negative_per_model[m][-1] = \
                                    model_summ.num_agreements_true_negative
                                summary.num_agreements_false_negative_per_model[m][-1] = \
                                    model_summ.num_agreements_false_negative

                            if not parsed_args.disable_ui:
                                if t % UPDATE_SPEED_FREQUENCY_EVAL == 0:
                                    curr_time = time.time()
                                    timestep_speed = t / (curr_time - timestep_speed_measure_time_start)

                            t += 1

    if PROFILE:
        # noinspection PyUnboundLocalVariable
        pr.disable()
        print("")
        pr.print_stats(sort='cumtime')

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ": C-Cure finished running.")
