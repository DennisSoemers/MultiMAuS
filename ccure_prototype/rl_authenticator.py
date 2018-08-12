"""
This module implements a Reinforcement Learning-based authenticator for the MultiMAuS simulator.
Hopefully it will require few changes in the simulator's own code, but some will probably be inevitable.

@author Dennis Soemers
"""

import numpy as np
import pandas as pd

from authenticators.abstract_authenticator import AbstractAuthenticator
from collections import defaultdict


class CSModelPerformanceSummary:

    def __init__(self):
        self.num_true_positives = 0
        self.num_false_positives = 0
        self.num_true_negatives = 0
        self.num_false_negatives = 0
        self.total_fraud_amounts_detected = 0.0
        self.num_agreements = 0
        self.num_agreements_true_positive = 0
        self.num_agreements_false_positive = 0
        self.num_agreements_true_negative = 0
        self.num_agreements_false_negative = 0

        self.num_true_positive_bins = [0, ]
        self.num_false_positive_bins = [0, ]
        self.num_true_negative_bins = [0, ]
        self.num_false_negative_bins = [0, ]
        self.num_agreements_bins = [0, ]
        self.num_agreements_true_positive_bins = [0, ]
        self.num_agreements_false_positive_bins = [0, ]
        self.num_agreements_true_negative_bins = [0, ]
        self.num_agreements_false_negative_bins = [0, ]


class RLAuthenticator(AbstractAuthenticator):

    def __init__(self, agent, state_creator, flat_fee, relative_fee, cs_model_config_names, simulator):
        super().__init__()

        self.agent = agent
        self.state_creator = state_creator
        self.flat_fee = flat_fee
        self.relative_fee = relative_fee

        self.secondary_auths_per_card = defaultdict(int)
        self.rejected_secondary_auths_per_card = defaultdict(int)
        self.num_unauthenticated_transactions_in_a_row_per_card = defaultdict(int)
        self.successful_trans_per_card = defaultdict(int)
        self.avg_reward_per_trans = defaultdict(float)

        self.num_trans_attempts_per_card = defaultdict(int)
        self.trans_count_bins = [0, ]
        self.second_auth_count_bins = [0, ]

        self.num_secondary_auths = 0
        self.num_secondary_auths_blocked_frauds = 0
        self.num_secondary_auths_blocked_genuines = 0
        self.num_secondary_auths_passed_genuines = 0

        self.total_fraud_amounts_seen = 0.0

        self.cs_model_config_names = cs_model_config_names
        self.cs_model_performance_summaries = \
            {model_name: CSModelPerformanceSummary() for model_name in cs_model_config_names}

        self.simulator = simulator

    def authorise_transaction(self, customer):
        state_features = self.state_creator.compute_state_vector_from_raw(
            customer,
            self.secondary_auths_per_card[customer.card_id],
            self.rejected_secondary_auths_per_card[customer.card_id],
            self.num_unauthenticated_transactions_in_a_row_per_card[customer.card_id],
            self.avg_reward_per_trans[customer.card_id]
        )

        scaled_state_features = self.scale_state_features(state_features)

        action = self.agent.choose_action_eps_greedy(
            scaled_state_features,
            customer.card_id,
            t=customer.model.schedule.time,
            transaction_date=customer.model.curr_global_date.replace(tzinfo=None),
            fraud=customer.fraudster
        )

        self.num_trans_attempts_per_card[customer.card_id] += 1
        num_trans_attempts = self.num_trans_attempts_per_card[customer.card_id]

        # unless there's a bug, the following "while" should really be equivalent to an if
        assert num_trans_attempts <= len(self.trans_count_bins)

        while num_trans_attempts >= len(self.trans_count_bins):
            self.trans_count_bins.append(0)
            self.second_auth_count_bins.append(0)

        self.trans_count_bins[num_trans_attempts] += 1

        success = True

        if action == 1:
            # ask secondary authentication
            authentication = customer.give_authentication()
            self.num_secondary_auths += 1

            self.secondary_auths_per_card[customer.card_id] += 1
            self.num_unauthenticated_transactions_in_a_row_per_card[customer.card_id] = 0

            self.second_auth_count_bins[num_trans_attempts] += 1

            if authentication is None:
                # customer refused to authenticate --> reward = 0
                reward = 0
                success = False
                self.rejected_secondary_auths_per_card[customer.card_id] += 1

                if customer.fraudster:
                    self.num_secondary_auths_blocked_frauds += 1
                else:
                    self.num_secondary_auths_blocked_genuines += 1
            else:
                # successful secondary authentication, so we know for sure it's a genuine transaction
                reward = self.compute_fees(state_features[1])
                self.num_secondary_auths_passed_genuines += 1

                curr_avg_reward = self.avg_reward_per_trans[customer.card_id]
                curr_num_trans = self.successful_trans_per_card[customer.card_id]

                self.avg_reward_per_trans[customer.card_id] = \
                    ((curr_avg_reward * curr_num_trans) + reward) / (curr_num_trans + 1)

                self.successful_trans_per_card[customer.card_id] += 1
        else:
            # did not ask for secondary authentication, will by default assume always a successful genuine transaction
            reward = self.compute_fees(state_features[1])

            self.num_unauthenticated_transactions_in_a_row_per_card[customer.card_id] += 1
            self.successful_trans_per_card[customer.card_id] += 1

        self.agent.register_reward(reward, customer.card_id)

        # some book-keeping
        cs_model_preds = state_features[7:]

        if customer.fraudster:
            self.total_fraud_amounts_seen += state_features[1]

            for i in range(len(self.cs_model_config_names)):
                summ = self.cs_model_performance_summaries[self.cs_model_config_names[i]]

                assert cs_model_preds[i] == 0 or cs_model_preds[i] == 1

                if cs_model_preds[i] == 0:
                    summ.num_false_negatives += 1

                    self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_false_negative_bins)

                    if action == 0:
                        summ.num_agreements += 1
                        summ.num_agreements_false_negative += 1

                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_bins)
                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_false_negative_bins)
                else:
                    summ.num_true_positives += 1
                    summ.total_fraud_amounts_detected += state_features[1]

                    self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_true_positive_bins)

                    if action == 1:
                        summ.num_agreements += 1
                        summ.num_agreements_true_positive += 1

                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_bins)
                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_true_positive_bins)
        else:
            for i in range(len(self.cs_model_config_names)):
                summ = self.cs_model_performance_summaries[self.cs_model_config_names[i]]

                if cs_model_preds[i] == 0:
                    summ.num_true_negatives += 1

                    self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_true_negative_bins)

                    if action == 0:
                        summ.num_agreements += 1
                        summ.num_agreements_true_negative += 1

                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_bins)
                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_true_negative_bins)
                else:
                    summ.num_false_positives += 1

                    self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_false_positive_bins)

                    if action == 1:
                        summ.num_agreements += 1
                        summ.num_agreements_false_positive += 1

                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_bins)
                        self.increment_summ_bin(bin_idx=num_trans_attempts, list_of_bins=summ.num_agreements_false_positive_bins)

        return success

    def compute_fees(self, amount):
        return self.flat_fee + self.relative_fee * amount

    def increment_summ_bin(self, bin_idx, list_of_bins):
        while bin_idx >= len(list_of_bins):
            list_of_bins.append(0)

        list_of_bins[bin_idx] += 1

    def scale_state_features(self, state_features):
        state_features = np.copy(state_features)

        state_features[1] /= self.simulator.max_abs_transaction_amount
        state_features[6] /= self.simulator.max_abs_transaction_amount

        state_features[3] /= self.simulator.max_num_trans_single_card
        state_features[4] /= self.simulator.max_num_trans_single_card
        state_features[5] /= self.simulator.max_num_trans_single_card

        state_features[2] /= self.simulator.max_num_timesteps_between_trans

        return state_features

    def update_fraudulent(self, transaction):
        """
        Updates the RL agent with new information that the given transaction turns out to be fraudulent.
        This transaction will have been used for an update where it was assumed to be genuine in the past (with
        a positive reward). So, in this update, we take the true negative reward of a fraudulent transaction, but
        ALSO subtract the reward that we previously mistakenly added.

        This can only be called for transactions that did not go through secondary authentication, so we know
        the action that we need to update for will always be 0.

            TODO probably want to rethink what we do with eligibility traces here.
            or maybe not? if a fraudulent transaction got through, that same customer would probably wish
            we'd been a bit more strict in general, also for all of his genuine transactions?

        :param transaction:
        :return:
        """
        transaction_df = pd.DataFrame([transaction])
        df_with_features = self.state_creator.feature_processing_func(transaction_df)
        transaction_row = df_with_features.iloc[0]
        state_features = self.state_creator.compute_state_vector_from_features(
            transaction_row,
            self.secondary_auths_per_card[transaction.CardID],
            self.rejected_secondary_auths_per_card[transaction.CardID],
            self.num_unauthenticated_transactions_in_a_row_per_card[transaction.CardID],
            self.avg_reward_per_trans[transaction.CardID]
        )

        scaled_state_features = self.scale_state_features(state_features)

        # we lose (= negative reward) the full transaction amount, and the fees we previously mistakenly assumed to
        # have been rewards
        reward = -(state_features[1] + self.compute_fees(state_features[1]))

        curr_avg_card_reward = self.avg_reward_per_trans[transaction.CardID]
        num_card_trans = self.successful_trans_per_card[transaction.CardID]
        self.avg_reward_per_trans[transaction.CardID] = \
            ((curr_avg_card_reward * num_card_trans) + reward) / num_card_trans

        if self.agent.is_card_id_known(transaction.CardID):
            # this should always be true, but appears to be very rarely false
            if transaction.Global_Date == self.agent.get_last_date(transaction.CardID):

                # in this case we can simply modify the most recently registered reward, we didn't learn from it yet
                self.agent.reset_fraud_reward(card_id=transaction.CardID, reward=-state_features[1])
            else:
                self.agent.fake_learn(
                    state_features=scaled_state_features,
                    action=0,
                    card_id=transaction.CardID,
                    t=self.simulator.schedule.time,
                    reward=reward
                )

                # immediately take another learning step with an inactive state,
                # so that we can learn from the discovered fraud ASAP (don't wait until we actually reach a new state)
                self.agent.fake_learn(
                    state_features=self.scale_state_features(self.state_creator.compute_inactive_state(
                        transaction.CardID,
                        0,
                        self
                    )),
                    action=2,
                    card_id=transaction.CardID,
                    t=self.simulator.schedule.time,
                    reward=0.0
                )
        else:
            print("WARNING: cannot update fraudulent case, card ID {} not known to RL agent!".format(transaction.CardID))


class StateCreator:

    def __init__(self, make_predictions_func, feature_processing_func, num_models):
        self.make_predictions_func = make_predictions_func
        self.feature_processing_func = feature_processing_func

        self.num_state_features = 7 + num_models

    def compute_state_vector_from_raw(
            self, customer,
            num_secondary_auths_card_id,
            num_rejected_secondary_auths_card_id,
            num_unauthenticated_transactions_card_id,
            avg_reward_per_trans
    ):
        """
        Uses the given customer's current properties to create a feature vector.

        Due to existing implementation of the MultiMAuS simulator, this is a little bit messy. First need to
        construct a single-row dataframe, pass that into the feature engineering classes, and then extract
        the processed feature vector which can be used by trained models to make predictions.

        Some code duplication here from the log_collector used by the simulator to generate data logs
        """
        multi_index = pd.MultiIndex.from_arrays([[0], [customer.unique_id]], names=["Step", "AgentID"])
        raw_transaction_df = pd.DataFrame({
            "Global_Date": pd.Series([customer.model.curr_global_date.replace(tzinfo=None)], dtype='datetime64[ns]',
                                     index=multi_index),
            "Local_Date": pd.Series([customer.local_datetime.replace(tzinfo=None)], dtype='datetime64[ns]',
                                    index=multi_index),
            "CardID": pd.Series([customer.card_id], dtype='int64',
                                index=multi_index),
            "MerchantID": pd.Series([customer.curr_merchant.unique_id], dtype='int64',
                                    index=multi_index),
            "Amount": pd.Series([customer.curr_amount], dtype='float64',
                                index=multi_index),
            "Currency": pd.Series([customer.currency], dtype='object',
                                  index=multi_index),
            "Country": pd.Series([customer.country], dtype='object',
                                 index=multi_index),
            "Target": pd.Series([customer.fraudster], dtype='int8',
                                index=multi_index),
            "AuthSteps": pd.Series([customer.curr_auth_step], dtype='int64',
                                   index=multi_index),
            "TransactionCancelled": pd.Series([customer.curr_trans_cancelled], dtype='bool',
                                              index=multi_index),
            "TransactionSuccessful": pd.Series([not customer.curr_trans_cancelled], dtype='bool',
                                               index=multi_index),
            "TimeSinceLastTransaction": pd.Series([customer.time_since_last_transaction], dtype='int64',
                                                  index=multi_index),
            "Timestamp": pd.Series([(customer.model.curr_global_date.replace(tzinfo=None)
                                     - customer.model.start_global_date).total_seconds() / 3600], dtype='int64',
                                   index=multi_index),
        })

        # use functor to add useful features to this single-row dataframe
        df_with_features = self.feature_processing_func(raw_transaction_df)
        transaction_row = df_with_features.iloc[0]

        # we don't want to use all of the features above for the Reinforcement Learner, but we do want to pass
        # them into offline trained models and use their outputs as features
        return self.compute_state_vector_from_features(
            transaction_row,
            num_secondary_auths_card_id,
            num_rejected_secondary_auths_card_id,
            num_unauthenticated_transactions_card_id,
            avg_reward_per_trans
        )

    def compute_state_vector_from_features(
            self, feature_vector,
            num_secondary_auths_card_id,
            num_rejected_secondary_auths_card_id,
            num_unauthenticated_transactions_card_id,
            avg_reward_per_trans
    ):
        """
        Computes state vector from a vector with features
        """
        state_features = []

        state_features.append(1.0)  # intercept                                     0
        state_features.append(feature_vector.Amount)                            #   1
        state_features.append(feature_vector.TimeSinceLastTransaction)          #   2
        state_features.append(num_secondary_auths_card_id)                      #   3
        state_features.append(num_rejected_secondary_auths_card_id)             #   4
        state_features.append(num_unauthenticated_transactions_card_id)         #   5
        state_features.append(avg_reward_per_trans)                             #   6

        predictions = self.make_predictions_func(feature_vector.values)
        #print(predictions)
        state_features.extend(predictions.flatten().tolist())

        #if not self.num_state_features == len(state_features):
        #    print(predictions)
        #    print(state_features)

        assert self.num_state_features == len(state_features)
        return np.array(state_features)

    def compute_inactive_state(
            self,
            card_id,
            time_since_last_transaction,
            authenticator
    ):
        state_features = []

        state_features.append(1.0)  # intercept
        state_features.append(0.0)  # Amount
        state_features.append(time_since_last_transaction)
        state_features.append(authenticator.secondary_auths_per_card[card_id])
        state_features.append(authenticator.rejected_secondary_auths_per_card[card_id])
        state_features.append(authenticator.num_unauthenticated_transactions_in_a_row_per_card[card_id])
        state_features.append(authenticator.avg_reward_per_trans[card_id])

        predictions = [0.0] * (self.num_state_features - len(state_features))
        state_features.extend(predictions)

        assert self.num_state_features == len(state_features)
        return np.array(state_features)

    def get_num_state_features(self):
        return self.num_state_features
