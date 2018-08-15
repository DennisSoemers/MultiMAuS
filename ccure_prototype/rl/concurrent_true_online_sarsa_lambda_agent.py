"""

@author Dennis Soemers
"""

import numpy as np

from simulator.customers import GenuineCustomer
from simulator.customers import FraudulentCustomer

from icecream import ic


class CustomerData:
    """
    Class for objects containing customer-specific data that we need for our Concurrent RL
    """

    def __init__(self, dim, global_theta, t, state_action_features, transaction_date):
        self.V = np.eye(dim)
        self.theta = np.copy(global_theta)
        self.theta_old = np.copy(global_theta)
        self.e = np.zeros(dim)
        self.tau = t
        self.tau_old = 0
        self.H = np.zeros((dim, dim))
        self.R_sum = 0.0
        self.phi = np.copy(state_action_features)
        self.last_transaction_date = transaction_date


class ConcurrentTrueOnlineSarsaLambdaAgent:

    def __init__(self, num_real_actions, num_actions, num_state_features,
                 gamma=0.999, base_alpha=0.025, lambda_=0.8):
        self.dim = num_state_features * num_actions
        self.num_real_actions = num_real_actions
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = base_alpha / self.dim
        self.lambda_ = lambda_

        self.theta = np.zeros(self.dim)   # TODO different initialization
        #self.theta[num_state_features:2*num_state_features] = 1.0   # this initialization encourages secondary authentications
        self.customer_data_map = {}     # for every customer, an object with customer-specific data such as traces

        self.genuine_stolen_ids_left = set()
        self.fraudster_stolen_ids_left = set()

        # following currently only tracked in cases where we exploit, not in cases where we explore
        self.genuine_q_values_no_auth = list()
        self.genuine_q_values_auth = list()
        self.fraud_q_values_no_auth = list()
        self.fraud_q_values_auth = list()

        self.first_action_time = None

        self.fixed_actions_map = {}

    def choose_action_eps_greedy(self, state_features, card_id, t, transaction_date, fraud, epsilon=0.1):
        if self.first_action_time is None:
            self.first_action_time = t  # need to account for high starting time due to generation of training data etc.

        epsilon = max(epsilon, (600.0 - (t - self.first_action_time)) / 600.0)

        if np.random.random_sample() < epsilon:
            action = int(np.random.random_sample() * self.num_real_actions)

            # TODO get rid of this if it doesn't work well
            self.fixed_actions_map[card_id] = action
        else:
            best_q = -1_000_000.0
            best_actions = []

            # we'll treat actions outside of num__real_actions as NULL action, never selected when not forced to
            for a in range(self.num_real_actions):
                x = self.state_action_feature_vector(state_features, a)

                q = np.dot(self.theta, x)

                if q > best_q:
                    best_q = q
                    best_actions = [a, ]
                elif q == best_q:
                    best_actions.append(a)

                if a == 0:
                    if fraud:
                        self.fraud_q_values_no_auth.append(q)
                    else:
                        self.genuine_q_values_no_auth.append(q)
                else:
                    if fraud:
                        self.fraud_q_values_auth.append(q)
                    else:
                        self.genuine_q_values_auth.append(q)

            #ic(best_actions)

            idx = int(np.random.random_sample() * len(best_actions))

            #ic(idx)
            #ic(best_q)

            action = best_actions[idx]

        # TODO probably remove this again?
        if card_id in self.fixed_actions_map:
            action = self.fixed_actions_map[card_id]

        if card_id not in self.customer_data_map:
            # first action for this customer, initialize customer
            self.customer_data_map[card_id] = CustomerData(
                dim=self.dim,
                global_theta=self.theta,
                t=t,
                state_action_features=self.state_action_feature_vector(state_features, action),
                transaction_date=transaction_date
            )
        else:
            # not the first time we take an action for this customer, so we can take a learning step
            self.customer_data_map[card_id].last_transaction_date = transaction_date
            self.learn(phi_prime=self.state_action_feature_vector(state_features, action), card_id=card_id, t=t)

        return action

    def get_last_date(self, card_id):
        return self.customer_data_map[card_id].last_transaction_date

    def get_weights(self):
        return self.theta

    def fake_learn(self, state_features, action, card_id, t, reward, terminal=False, customer=None):
        if terminal and customer is not None:
            # only take the terminal learning step if there isn't another user in the system with the
            # same card ID (happens in case of stolen card IDs)
            if isinstance(customer, GenuineCustomer) and customer.card_stolen:
                if card_id not in self.fraudster_stolen_ids_left:
                    return      # fraudster who stole card didn't leave yet, so wait, don't learn
            elif isinstance(customer, FraudulentCustomer) and customer.stole_card:
                if card_id not in self.genuine_stolen_ids_left:
                    return      # genuine card holder didn't leave yet, so wait, don't learn

        self.learn(
            phi_prime=self.state_action_feature_vector(state_features, action),
            card_id=card_id,
            t=t,
            terminal=terminal
        )

        self.register_reward(R=reward, card_id=card_id)

    def is_card_id_known(self, card_id):
        return card_id in self.customer_data_map

    def learn(self, phi_prime, card_id, t, terminal=False):
        customer_data = self.customer_data_map[card_id]

        if terminal:
            # expect 0 future reward
            phi_prime = np.zeros(self.dim)

        psi = self.theta - customer_data.theta
        phi = customer_data.phi

        delta = customer_data.R_sum + self.gamma**(t + 1 - customer_data.tau) * np.dot(self.theta, phi_prime) \
                - np.dot(customer_data.theta_old + psi, phi)

        alpha_phi_phi = self.alpha * np.outer(phi, phi)
        gamma_tau_tau_old = self.gamma**(customer_data.tau - customer_data.tau_old)
        dot_e_phi = np.dot(customer_data.e, phi)

        customer_data.V = np.matmul((np.eye(self.dim) - alpha_phi_phi), customer_data.V)

        customer_data.H = customer_data.H - np.matmul(alpha_phi_phi, customer_data.H) \
                          + gamma_tau_tau_old * np.outer(customer_data.e, phi) \
                          - gamma_tau_tau_old * dot_e_phi * alpha_phi_phi

        customer_data.e = self.lambda_ * gamma_tau_tau_old * customer_data.e \
                          + self.alpha * phi \
                          - self.alpha * self.lambda_ * gamma_tau_tau_old * dot_e_phi * phi

        theta_new = customer_data.theta + delta * customer_data.e \
                    + self.alpha * np.dot(customer_data.theta_old + psi - customer_data.theta, phi) * phi \
                    + np.matmul(customer_data.V + (1.0 - self.lambda_) * customer_data.H, psi) \
                    + np.dot(psi, phi) * (customer_data.e - self.alpha * phi)

        customer_data.theta_old = np.copy(self.theta)
        self.theta = theta_new
        customer_data.theta = np.copy(self.theta)

        customer_data.phi = phi_prime
        customer_data.R_sum = 0.0
        customer_data.tau_old = customer_data.tau
        customer_data.tau = t + 1

    def on_customer_leave(self, card_id, customer):
        if card_id in self.customer_data_map:
            if isinstance(customer, GenuineCustomer) and customer.card_stolen:
                if card_id in self.fraudster_stolen_ids_left:
                    self.customer_data_map.pop(card_id)
                    self.fraudster_stolen_ids_left.remove(card_id)
                else:
                    self.genuine_stolen_ids_left.add(card_id)
            elif isinstance(customer, FraudulentCustomer) and customer.stole_card:
                if card_id in self.genuine_stolen_ids_left:
                    self.customer_data_map.pop(card_id)
                    self.genuine_stolen_ids_left.remove(card_id)
                else:
                    self.fraudster_stolen_ids_left.add(card_id)
            else:
                self.customer_data_map.pop(card_id)

    def register_reward(self, R, card_id):
        self.customer_data_map[card_id].R_sum += R

    def reset_fraud_reward(self, card_id, reward):
        self.customer_data_map[card_id].R_sum = reward

    def state_action_feature_vector(self, state_features, action):
        """
        Creates a state-action feature vector from a given vector of state features and a given action

        :param state_features:
            Feature vector for state
        :param action:
            Action (integer)
        :return:
            State-action feature vector
        """
        x = []
        for a in range(self.num_actions):
            if a == action:
                x = np.append(x, state_features)
            else:
                x = np.append(x, np.zeros(state_features.size))

        return x
