from authenticators.abstract_authenticator import AbstractAuthenticator


class OracleAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        if customer.fraudster:
            customer.give_authentication()
            return False
        else:
            return True


class NeverSecondAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        return True


class AlwaysSecondAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        if customer.give_authentication() is not None:
            return True
        else:
            return False


class HeuristicAuthenticator(AbstractAuthenticator):
    def __init__(self, thresh=50):
        super().__init__()
        self.thresh = thresh

    def authorise_transaction(self, customer):
        if customer.curr_amount > self.thresh:
            if customer.give_authentication() is not None:
                return True
            else:
                return False

    def take_action(self, customer):
        if customer.curr_amount > self.thresh:
            return 1


class RandomAuthenticator(AbstractAuthenticator):
    def authorise_transaction(self, customer):
        # ask for second authentication in 50% of the cases
        if customer.model.random_state.uniform(0, 1, 1)[0] < 0.5:
            if customer.give_authentication() is not None:
                return True
            else:
                return False
