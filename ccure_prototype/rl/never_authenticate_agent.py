"""

@author Dennis Soemers
"""

import numpy as np

from simulator.customers import GenuineCustomer
from simulator.customers import FraudulentCustomer


class NeverAuthenticateAgent:

    def __init__(self, num_actions, num_state_features):
        self.dim = num_state_features * num_actions
        self.theta = np.zeros(self.dim)   # TODO different initialization

        self.genuine_stolen_ids_left = set()
        self.fraudster_stolen_ids_left = set()

        # following currently only tracked in cases where we exploit, not in cases where we explore
        self.genuine_q_values_no_auth = list()
        self.genuine_q_values_auth = list()
        self.fraud_q_values_no_auth = list()
        self.fraud_q_values_auth = list()

    def choose_action_eps_greedy(self, state_features, card_id, t, transaction_date, fraud, epsilon=0.1):
        return 0

    def get_last_date(self, card_id):
        return None

    def get_weights(self):
        return self.theta

    def fake_learn(self, state_features, action, card_id, t, reward, terminal=False, customer=None):
        pass

    def is_card_id_known(self, card_id):
        return False

    def learn(self, phi_prime, card_id, t, terminal=False):
        pass

    def on_customer_leave(self, card_id, customer):
        pass

    def register_reward(self, R, card_id):
        pass

    def reset_fraud_reward(self, card_id, reward):
        pass

    def state_action_feature_vector(self, state_features, action):
        return None
