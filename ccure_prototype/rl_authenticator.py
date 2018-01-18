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

    def authorise_transaction(self, customer):
        state_features = self.state_creator.compute_state_vector_from_raw(customer)
        action = self.agent.choose_action_eps_greedy(state_features)
        success = True

        if action == 1:
            # ask secondary authentication
            authentication = customer.give_authentication()

            if authentication is None:
                # customer refused to authenticate --> reward = 0
                reward = 0
                success = False
            else:
                # successful secondary authentication, so we know for sure it's a genuine transaction
                reward = self.compute_fees(state_features[1])
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
        state_features = self.state_creator.compute_state_vector_from_features(transaction)

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

    def __init__(self, trained_models, feature_processing_func):
        self.trained_models = trained_models
        self.feature_processing_func = feature_processing_func

        self.num_state_features = 3  # TODO properly compute this

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

        # start with code based on the log_collector implementation to generate raw-feature dataframe
        # this is probably way too much and too complicated code, but it works
        agent_reporters = {
            "Global_Date": lambda c: c.model.curr_global_date.replace(tzinfo=None),
            "Local_Date": lambda c: c.local_datetime.replace(tzinfo=None),
            "CardID": lambda c: c.card_id,
            "MerchantID": lambda c: c.curr_merchant.unique_id,
            "Amount": lambda c: c.curr_amount,
            "Currency": lambda c: c.currency,
            "Country": lambda c: c.country,
            "Target": lambda c: c.fraudster,
            "AuthSteps": lambda c: c.curr_auth_step,
            "TransactionCancelled": lambda c: c.curr_trans_cancelled,
            "TransactionSuccessful": lambda c: not c.curr_trans_cancelled,
            "TimeSinceLastTransaction": lambda c: c.time_since_last_transaction
        }

        agent_vars = {}
        for var, reporter in agent_reporters.items():
            agent_records = [(customer.unique_id, reporter(customer))]

            if var not in agent_vars:
                agent_vars[var] = []

            agent_vars[var].append(agent_records)

        data = defaultdict(dict)

        for var, records in agent_vars.items():
            for step, entries in enumerate(records):
                for entry in entries:
                    agent_id = entry[0]
                    val = entry[1]
                    data[(step, agent_id)][var] = val

        raw_transaction_df = pd.DataFrame.from_dict(data, orient="index")
        raw_transaction_df.index.names = ["Step", "AgentID"]

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

        assert self.num_state_features == len(state_features)
        return np.array(state_features)

    def get_num_state_features(self):
        return self.num_state_features
