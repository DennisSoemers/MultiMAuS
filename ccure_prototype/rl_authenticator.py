"""
This module implements a Reinforcement Learning-based authenticator for the MultiMAuS simulator.
Hopefully it will require few changes in the simulator's own code, but some will probably be inevitable.

@author Dennis Soemers
"""

import numpy as np
import pandas as pd

from collections import defaultdict


class RLAuthenticator:

    def __init__(self, agent, state_creator):
        self.agent = agent
        self.state_creator = state_creator


class StateCreator:

    def __init__(self, trained_models, feature_processing_func):
        self.trained_models = trained_models
        self.feature_processing_func = feature_processing_func

        self.num_state_features = 0  # TODO properly compute this

    def compute_feature_vector(self, customer):
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
            "TransactionSuccessful": lambda c: not c.curr_trans_cancelled
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

        # we don't want to use all of the features above for the Reinforcement Learner, but we do want to pass
        # them into offline trained models and use their outputs as features
        state_features = []

        # TODO append features to the above list

        return np.array(state_features)

    def get_num_state_features(self):
        return self.num_state_features
