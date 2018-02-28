"""
This module implements a Reinforcement Learning-based authenticator for the MultiMAuS simulator.
Hopefully it will require few changes in the simulator's own code, but some will probably be inevitable.

@author Dennis Soemers
"""

import numpy as np
import pandas as pd

from authenticators.abstract_authenticator import AbstractAuthenticator
from collections import defaultdict


class RLAuthenticator(AbstractAuthenticator):

    def __init__(self, agent, state_creator, flat_fee, relative_fee):
        self.agent = agent
        self.state_creator = state_creator
        self.flat_fee = flat_fee
        self.relative_fee = relative_fee

        self.num_secondary_auths = 0
        self.num_secondary_auths_blocked_frauds = 0
        self.num_secondary_auths_blocked_genuines = 0
        self.num_secondary_auths_passed_genuines = 0

    def authorise_transaction(self, customer):
        state_features = self.state_creator.compute_state_vector_from_raw(customer)
        action = self.agent.choose_action_eps_greedy(state_features)
        success = True

        if action == 1:
            # ask secondary authentication
            authentication = customer.give_authentication()
            self.num_secondary_auths += 1

            if authentication is None:
                # customer refused to authenticate --> reward = 0
                reward = 0
                success = False

                if customer.fraudster:
                    self.num_secondary_auths_blocked_frauds += 1
                else:
                    self.num_secondary_auths_blocked_genuines += 1
            else:
                # successful secondary authentication, so we know for sure it's a genuine transaction
                reward = self.compute_fees(state_features[1])
                self.num_secondary_auths_passed_genuines += 1
        else:
            # did not ask for secondary authentication, will by default assume always a successful genuine transaction
            reward = self.compute_fees(state_features[1])

        time_since_last_transaction = state_features[2]
        if time_since_last_transaction > 0:
            # divide reward by time since last transaction, because rewards in episodes with infrequent rewards are
            # less important than rewards in episodes with frequent rewards
            reward /= time_since_last_transaction

        # take a learning step
        self.agent.learn(state_features, action, reward, customer.card_id)

        return success

    def compute_fees(self, amount):
        return self.flat_fee + self.relative_fee * amount

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
        df_with_features = self.feature_processing_func(transaction_df)
        transaction_row = df_with_features.iloc[0]
        state_features = self.state_creator.compute_state_vector_from_features(transaction_row)

        # we lose (= negative reward) the full transaction amount, and the fees we previously mistakenly assumed to
        # have been rewards
        reward = -(state_features[1] + self.compute_fees(state_features[1]))

        time_since_last_transaction = state_features[2]
        if time_since_last_transaction > 0:
            # divide reward by time since last transaction, because rewards in episodes with infrequent rewards are
            # less important than rewards in episodes with frequent rewards
            reward /= time_since_last_transaction

        self.agent.learn(state_features=state_features, action=0, reward=reward, card_id=transaction.CardID)


class StateCreator:

    def __init__(self, make_predictions_func, feature_processing_func, num_models):
        self.make_predictions_func = make_predictions_func
        self.feature_processing_func = feature_processing_func

        self.num_state_features = 3 + num_models

    def compute_state_vector_from_raw(self, customer):
        """
        Uses the given customer's current properties to create a feature vector.

        Due to existing implementation of the MultiMAuS simulator, this is a little bit messy. First need to
        construct a single-row dataframe, pass that into the feature engineering classes, and then extract
        the processed feature vector which can be used by trained models to make predictions.

        Some code duplication here from the log_collector used by the simulator to generate data logs

        :param customer:
        :return:
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
        return self.compute_state_vector_from_features(transaction_row)

    def compute_state_vector_from_features(self, feature_vector):
        """
        Computes state vector from a vector with features

        :param feature_vector:
        :return:
        """
        state_features = []

        state_features.append(1.0)  # intercept
        state_features.append(feature_vector.Amount)
        state_features.append(feature_vector.TimeSinceLastTransaction)

        predictions = self.make_predictions_func(feature_vector.values)
        #print(predictions)
        state_features.extend(predictions.flatten().tolist())

        assert self.num_state_features == len(state_features)
        return np.array(state_features)

    def get_num_state_features(self):
        return self.num_state_features
