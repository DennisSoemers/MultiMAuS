"""
This script runs a full prototype for the C-Cure project

Info on the C-Cure project: http://www.securit-brussels.be/project/c-cure/

Main author for this script:
    @author Dennis Soemers (VUB AI Lab)

Many other people also contributed to the project from
Vrije Universiteit Brussel (AI Lab and BUTO) and Université Catholique de Louvain (Crypto Group)
"""

import atexit
import random
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
from ccure_prototype.rl.true_online_sarsa_lambda_agent import TrueOnlineSarsaLambdaAgent
from data.features.aggregate_features import AggregateFeatures
from data.features.apate_graph_features import ApateGraphFeatures
from simulator import parameters
from simulator.transaction_model import TransactionModel


if __name__ == '__main__':
    # want stack trace when warnings are raised
    np.seterr(all='raise')

    logger = multiprocessing.log_to_stderr(logging.INFO)


    @atexit.register
    def notify():
        """
        Make some noise when we crash / terminate (ONLY WORKS ON WINDOWS)
        """
        import winsound
        winsound.Beep(440, 500)


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
    #NUM_SIM_STEPS_TRAINING_DATA = 20_000
    NUM_SIM_STEPS_TRAINING_DATA = 4000
    # ratio out of training data to use for feature engineering (necessary to learn how to create APATE graph features)
    TRAIN_FEATURE_ENGINEERING_RATIO = 0.125

    # number of steps to simulate and discard afterwards to create gap between training and test data
    #NUM_SIM_STEPS_SKIP = 5_000
    NUM_SIM_STEPS_SKIP = 1000
    #NUM_SIM_STEPS_SKIP = 0

    # number of steps to simulate for evaluation
    NUM_SIM_STEPS_EVALUATION = 30_000
    #NUM_SIM_STEPS_EVALUATION = 200
    #NUM_SIM_STEPS_EVALUATION = 0

    # flat fee we take for every genuine transaction
    FLAT_FEE = 0.01
    # relative fee (multiplied by transaction amount) we take for every genuine transaction
    RELATIVE_FEE = 0.01

    # if True, we'll also profile our running code
    PROFILE = False

    # -------------------------------------------
    # R offline models
    # -------------------------------------------
    # filepath for offline models R script. NOTE: hardcoded
    CS_MODELS_R_FILEPATH = "D:/Apps/C-Cure/Code/R/CSModels.R"

    # -------------------------------------------
    # simulator parameters
    # -------------------------------------------
    simulator_params = parameters.get_default_parameters()
    simulator_params['end_date'] = datetime(9999, 12, 31)
    simulator_params['stay_prob'][0] = 0.9      # stay probability for genuine customers
    simulator_params['stay_prob'][1] = 0.99     # stay probability for fraudsters
    simulator_params['seed'] = random.randrange(2**32)
    #simulator_params['seed'] = 944577390
    seed = simulator_params['seed']
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ": Running C-Cure prototype with seed = ", seed)

    simulator_params['num_fraudsters'] = 300
    simulator_params['stay_prob'][0] = 0.9      # genuine
    simulator_params['stay_prob'][1] = 0.96     # fraud

    # we assume fraudulent cases get reported after 6 simulator steps (hours)   TODO can make this much more fancy
    FRAUD_REPORT_TIME = 6

    # -------------------------------------------
    # output
    # -------------------------------------------
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'seed_{}'.format(simulator_params['seed']))

    results_run_idx = 0
    while True:
        if os.path.isdir(os.path.join(OUTPUT_DIR, "run{0:05d}".format(results_run_idx))):
            results_run_idx += 1
        else:
            break

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "run{0:05d}".format(results_run_idx))
    os.makedirs(OUTPUT_DIR)

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


    def on_customer_leave(card_id):
        if authenticator_test_phase is not None:
            authenticator_test_phase.agent.on_customer_leave(card_id)


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
    control_frame = create_control_frame()
    exit_simulation = False
    timestep_speed_measure_time_start = time.time()
    start_time_generate_training_data = timestep_speed_measure_time_start
    timestep_speed = 0

    # generate training data
    for t in range(NUM_SIM_STEPS_TRAINING_DATA):
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
        control_frame.update_info_generate_training_data(NUM_SIM_STEPS_TRAINING_DATA, NUM_SIM_STEPS_TRAINING_DATA,
                                                         timestep_speed,
                                                         total_time=time.time() - start_time_generate_training_data)
        control_frame.root.update()

        feature_learning_data = training_data.iloc[:num_feature_learning_instances]
        model_learning_data = training_data.iloc[num_feature_learning_instances:]

        # feature engineering

        # define feature engineering function that we can run in a different thread without blocking this one
        def feature_engineering_func(feature_learning_data, model_learning_data, out_list):
            global aggregate_feature_constructor
            global apate_graph_feature_constructor
            aggregate_feature_constructor = AggregateFeatures(feature_learning_data)
            apate_graph_feature_constructor = ApateGraphFeatures(feature_learning_data)
            update_feature_constructors_unlabeled(model_learning_data)

            # compute features for our model learning data
            model_learning_data = process_data(model_learning_data)

            # store in out_list so we can retrieve results in main thread
            out_list[0] = aggregate_feature_constructor
            out_list[1] = apate_graph_feature_constructor
            out_list[2] = model_learning_data

        # create new thread
        out_list = [None] * 3
        feature_eng_thread = threading.Thread(target=feature_engineering_func, name="Feature Eng. Thread",
                                              args=(feature_learning_data, model_learning_data, out_list),
                                              daemon=True)
        start_time_feature_engineering = time.time()
        feature_eng_thread.start()

        while feature_eng_thread.is_alive():
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
            control_frame.update_info_feature_engineering(time.time() - start_time_feature_engineering, finished=True)
            control_frame.root.update()

            # train offline models in R

            # create new process
            from ccure_prototype import r_model_training
            train_r_process = multiprocessing.Process(target=r_model_training.train_r_models,
                                                      name="R Model Training Process",
                                                      args=(CS_MODELS_R_FILEPATH,
                                                            OUTPUT_FILE_MODEL_LEARNING_DATA,
                                                            OUTPUT_DIR,
                                                            seed))
            start_time_r_model_training = time.time()
            train_r_process.daemon = True
            train_r_process.start()

            while train_r_process.is_alive():
                if not is_control_frame_alive():
                    exit_simulation = True
                    break

                # figure out for how many models we already have files
                trial_models_dir = os.path.join(OUTPUT_DIR, "TrialModels")
                final_models_dir = os.path.join(OUTPUT_DIR, "Models")

                if not os.path.isdir(trial_models_dir) or not os.path.isdir(final_models_dir):
                    num_models_trained = 0
                else:
                    num_models_trained = len(os.listdir(trial_models_dir)) + len(os.listdir(final_models_dir))

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
                trial_models_dir = os.path.join(OUTPUT_DIR, "TrialModels")
                final_models_dir = os.path.join(OUTPUT_DIR, "Models")

                control_frame.update_info_r_model_training(
                    len(os.listdir(trial_models_dir)) + len(os.listdir(final_models_dir)),
                    NUM_R_MODELS, time.time() - start_time_r_model_training)
                control_frame.root.update()

                # load R models into memory
                import rpy2.robjects as robjects
                robjects.r('set.seed({})'.format(seed))
                robjects.r('source(\"{}\")'.format(CS_MODELS_R_FILEPATH))
                savepath_string = OUTPUT_DIR.replace("\\", "/")
                if not savepath_string.endswith("/"):
                    savepath_string += "/"
                robjects.r('savepath<-\"{}\"'.format(savepath_string))
                num_r_predictions = int(robjects.r('loadCSModels(savepath)')[0])

                # create function that we can use to make predictions for transactions
                import rpy2.robjects.numpy2ri
                rpy2.robjects.numpy2ri.activate()

                def make_predictions(feature_vector):
                    return np.array(robjects.r.predictCSModels(feature_vector))

                # get rid of all fraudsters in training data and replace them with new fraudsters
                fraudster_ids = set()
                for transaction in training_data.itertuples():  # this loops through feature and model learning data at once
                    if transaction.Target == 1:
                        fraudster_ids.add(transaction.CardID)

                block_cards(list(fraudster_ids), replace_fraudsters=True)
                fraudster_ids = None    # clean memory

                # generate some data as a gap between training and test data
                start_time_generate_gap_data = time.time()

                for t in range(NUM_SIM_STEPS_SKIP):
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

                    if t % UPDATE_SPEED_FREQUENCY == 0:
                        curr_time = time.time()
                        timestep_speed = UPDATE_SPEED_FREQUENCY / (curr_time - timestep_speed_measure_time_start)
                        timestep_speed_measure_time_start = curr_time

                if not exit_simulation:
                    skip_data = get_log(clear_after=True)

                    # can still use the skip data to update our feature engineering
                    update_feature_constructors_unlabeled(skip_data)

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

                    # create our RL-based authenticator and add it to the simulator
                    state_creator = StateCreator(make_predictions_func=make_predictions,
                                                 feature_processing_func=process_data,
                                                 num_models=num_r_predictions)
                    authenticator_test_phase = RLAuthenticator(
                        agent=TrueOnlineSarsaLambdaAgent(
                            num_actions=2, num_state_features=state_creator.get_num_state_features()),
                        state_creator=state_creator, flat_fee=FLAT_FEE, relative_fee=RELATIVE_FEE)
                    simulator.authenticator = authenticator_test_phase

                    # in this dict, we'll store for simulation steps (key = t) lists of transactions that need to be reported
                    # at that time
                    to_report = {}

                    timestep_speed_measure_time_start = time.time()

                    with ExperimentSummary(flat_fee=FLAT_FEE,
                                           relative_fee=RELATIVE_FEE,
                                           output_dir=OUTPUT_DIR) as summary:

                        t = 0

                        while t < NUM_SIM_STEPS_EVALUATION:
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
