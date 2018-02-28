"""
True Online Sarsa(lambda) agent for deciding whether or not to request secondary authentication.
Pseudocode for the regular version of the algorithm can be found in Section 12.7 of the draft (5 November 2017)
of the second edition of Sutton and Barto's Reinforcement Learning book.

This implementation is a little different due to differences between the C-Cure setting and the standard
MDP formulation.

Our model:
- Every transaction is a state
- Every card ID is an ''episode'', where transactions for every card ID are states within that episode
- This means that we have lots of different episodes running concurrently

The regular update rule involves all of the following variables:

    S <-- current state
    A <-- action we take
    x <-- features for state-action pair (S, A)
    R <-- one-step reward from taking action A in state S
    S' <-- next state we move into
    A' <-- action we plan to take next in S'
    x' <-- features for state-action pair (S', A')

------------------------------------------------------------------------------------------------------------------
Expected Future Rewards
------------------------------------------------------------------------------------------------------------------

In our case, there's no problem with S, A, x. However, S' is problematic (and so are A' and x'
because they depend on S'). S' would have to be the next transaction by the same customer in our model.
It can take a really long time before our customer makes a new transaction, so we don't really want to wait
with our update until that happens. Even worse, it is possible that the customer never again makes a transaction,
so we may never be able to run such an update.

In the regular update rule, S' and A' are only used to construct x'. Then, x' is only used to compute Q',
which is our estimate of all future rewards if we continue following our policy. Because we don't want to wait
until we're able to compute x' (which may be never if a customer silently leaves), we'll instead simply always
take Q' = 0.

This results in correct updates in cases where a customer did indeed leave after we took our action, and updates
with a bias towards 0 in the term for expected future gains. Intuitively, I suspect this will cause us to mostly
focus on single-step rewards, and less on long-term rewards. To compensate for this, we should probably assign
rather high values to gamma and lambda (which also decay the influence of future rewards).

------------------------------------------------------------------------------------------------------------------
Immediate Rewards
------------------------------------------------------------------------------------------------------------------

The R (one-step / immediate reward) term can also be problematic for us in cases where we did not request secondary
authentication (transaction will go through, and we won't know if it was a genuine or a fraudulent transaction).
We will simply assume all such transactions are genuine initially, and therefore give positive rewards to the agent
for all transactions that go through. When transactions are later reported to have been fraudulent, we need support
for a ''reverse'' update of those fraudulent transactions, and then update them again with correct (negative) rewards.

@author Dennis Soemers
"""

import numpy as np


class TrueOnlineSarsaLambdaAgent:

    def __init__(self, num_actions, num_state_features,
                 gamma=0.99, base_alpha=0.1, _lambda=0.8, online_normalization=True):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = base_alpha / (num_actions * num_state_features)    # TODO decaying learning rate schedule
        self._lambda = _lambda

        self.weights = np.ones(num_actions * num_state_features)   # TODO different initialization
        self.z_map = {}     # for every customer a separate vector of dutch traces in this dictionary
        self.z_bounds_map = {}      # for every z vector, store a vector with most recently-used normalization bounds

        self.online_normalization = online_normalization
        self.feature_bounds = np.ones(num_actions * num_state_features)

    def choose_action_eps_greedy(self, state_features, epsilon=0.05):       # TODO decaying epsilon
        if np.random.random_sample() < epsilon:
            return int(np.random.random_sample() * self.num_actions)
        else:
            best_q = -1_000_000.0
            best_actions = []

            for a in range(self.num_actions):
                x = self.state_action_feature_vector(state_features, a)

                if self.online_normalization:
                    self.update_weights_normalization(x)
                    x = x / self.feature_bounds

                q = self.q_value(x)
                #print("Q({}) = {}".format(a, q))

                if q > best_q:
                    best_q = q
                    best_actions = [a, ]
                elif q == best_q:
                    best_actions.append(a)

            idx = int(np.random.random_sample() * len(best_actions))
            return best_actions[idx]

    def learn(self, state_features, action, reward, card_id):
        """
        Take a learning / update step. Note that we don't incorporate the next state
        and next action, as would normally be done in True Online Sarsa(lambda). See
        detailed comments at top of the file.

        :param state_features:
            Feature vector of state in which we were
        :param action:
            Action we took in the given state
        :param reward:
            Single-step reward
        :param card_id:
            Card ID of the customer (required for customer-specific dutch traces)
        """
        #print("learning from reward = ", reward)
        x = self.state_action_feature_vector(state_features, action)

        if self.online_normalization:
            self.update_weights_normalization(x)
            x = x / self.feature_bounds

        Q = self.q_value(x)
        delta = reward - Q  # in standard implementation of algorithm, this would include +gamma*Q', which we set to 0
        #print("Q = ", Q)
        #print("delta = ", delta)

        if card_id in self.z_map:
            z = self.z_map[card_id]

            if self.online_normalization:
                # update normalization of the z vector
                old_bounds = self.z_bounds_map[card_id]
                normalization_correction = old_bounds / self.feature_bounds
                z = np.multiply(z, normalization_correction)
        else:
            z = np.zeros(self.num_actions * len(state_features))

        z = self.gamma * self._lambda * z + \
            (1.0 - self.alpha * self.gamma * self._lambda * np.dot(z, x)) * x

        # the following line would normally twice include Q_old, which is always 0 in our case
        #print("updating weights from ", self.weights)
        self.weights = self.weights + self.alpha * (delta + Q) * z - self.alpha * Q * x
        #print("... to ", self.weights)
        #print("new Q estimate = ", self.q_value(x))
        #print("")

        self.z_map[card_id] = z
        self.z_bounds_map[card_id] = np.copy(self.feature_bounds)

    def on_customer_leave(self, card_id):
        if card_id in self.z_map:
            self.z_map.pop(card_id)
            self.z_bounds_map.pop(card_id)

    def q_value(self, x):
        """
        Computes Q-value for a state-action feature vector x

        :param x:
            State-action feature vector (num elements = num actions * num state features)
        :return:
            Q-value according to current weight vector
        """
        return np.dot(self.weights, x)

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

    def update_weights_normalization(self, x):
        """
        Updates weights vector according if necessary due to updated normalization given
        new state-action feature vector x
        """
        try:
            old_bounds = np.copy(self.feature_bounds)
            self.feature_bounds = np.maximum(old_bounds, np.abs(x))
            normalization_correction = old_bounds / self.feature_bounds
            self.weights = np.multiply(self.weights, normalization_correction)
        except:
            print(x)
            crash = 1/0
