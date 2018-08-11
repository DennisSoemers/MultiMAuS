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

    def __init__(self, t, state_action_features, transaction_date):
        self.stored_reward_sums = [0.0, ]
        self.stored_state_features = [state_action_features, ]
        self.stored_option_durations = [0, ]
        self.tau = t
        self.last_transaction_date = transaction_date


class NStepSarsaAgent:

    def __init__(self, num_real_actions, num_actions, num_state_features,
                 gamma=0.999, base_alpha=0.025, n=2, lambda_=0.8):
        self.dim = num_state_features * num_actions
        self.num_real_actions = num_real_actions
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = base_alpha / self.dim
        self.n = n

        self.theta = np.zeros(self.dim)   # TODO different initialization
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

            # we'll treat actions outside of num_real_actions as NULL action, never selected when not forced to
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

            idx = int(np.random.random_sample() * len(best_actions))
            action = best_actions[idx]

        # TODO probably remove this again?
        if card_id in self.fixed_actions_map:
            action = self.fixed_actions_map[card_id]

        if card_id not in self.customer_data_map:
            # first action for this customer, initialize customer
            self.customer_data_map[card_id] = CustomerData(
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

    def fake_learn(self, state_features, action, card_id, t, reward, terminal=False):
        self.learn(
            phi_prime=self.state_action_feature_vector(state_features, action),
            card_id=card_id,
            t=t,
            terminal=terminal
        )

        if not terminal:
            self.register_reward(R=reward, card_id=card_id)

    def is_card_id_known(self, card_id):
        return card_id in self.customer_data_map

    def learn(self, phi_prime, card_id, t, terminal=False):
        customer_data = self.customer_data_map[card_id]

        if terminal:
            # expect 0 future reward
            phi_prime = np.zeros(self.dim)

        # ------------------------------------------------------------------------------------------------------
        assert len(customer_data.stored_state_features) <= self.n
        assert len(customer_data.stored_option_durations) <= self.n
        assert len(customer_data.stored_reward_sums) <= self.n

        if len(customer_data.stored_state_features) == self.n:  # ran enough steps to compute full n-step returns
            G = 0.0
            extra_gamma_power = 0

            for i in range(self.n):
                G += (self.gamma ** extra_gamma_power) * customer_data.stored_reward_sums[i]
                extra_gamma_power += customer_data.stored_option_durations[i]

            if not terminal:
                G += (self.gamma ** extra_gamma_power) * np.dot(phi_prime, self.theta)

            # here's the actual update to parameters
            self.theta = self.theta + self.alpha * \
                         (G - np.dot(customer_data.stored_state_features[0], self.theta)) * customer_data.stored_state_features[0]

            # remove things at the front of our memory queues
            customer_data.stored_state_features.pop(0)
            customer_data.stored_reward_sums.pop(0)
            customer_data.stored_option_durations.pop(0)

        if terminal:
            # reached terminal state, so also need to run a bunch of updates with returns for < n steps
            while len(customer_data.stored_reward_sums) > 0:
                G = 0.0
                extra_gamma_power = 0

                for i in range(len(customer_data.stored_reward_sums)):
                    G += (self.gamma ** extra_gamma_power) * customer_data.stored_reward_sums[i]
                    extra_gamma_power += customer_data.stored_option_durations[i]

                # here's the actual update to parameters
                self.theta = self.theta + self.alpha * \
                             (G - np.dot(customer_data.stored_state_features[0], self.theta)) * customer_data.stored_state_features[0]

                # remove things at the front of our memory queues
                customer_data.stored_state_features.pop(0)
                customer_data.stored_reward_sums.pop(0)
                customer_data.stored_option_durations.pop(0)
        else:
            # start memorizing stuff for the next step
            customer_data.stored_state_features.append(phi_prime)
            customer_data.stored_reward_sums.append(0.0)
            customer_data.stored_option_durations.append(0)
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
        self.customer_data_map[card_id].stored_reward_sums[-1] += R
        self.customer_data_map[card_id].stored_option_durations[-1] += 1

    def reset_fraud_reward(self, card_id, reward):
        self.customer_data_map[card_id].stored_reward_sums[-1] = reward

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
