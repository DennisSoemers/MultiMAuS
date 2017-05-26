from datetime import datetime
from os.path import join
import numpy as np
from data import utils_data


def get_default_parameters():

    params = {

        # seed for random number generator of current simulation
        "seed": 666,

        # start and end date of simulation
        'start date': datetime(2016, 1, 1),
        'end date': datetime(2016, 12, 31),

        # max number of authentication steps (at least 1)
        "max authentication steps": 1,

        # number of customers and fraudsters at beginning of simulation
        "start num customers": 100,
        "start num fraudsters": 8,

        # number of merchants at the beginning of simulation
        "num merchants": 28,

        # amount range (in Euros)
        "min amount": 0.01,
        "max amount": 10500,

        # currencies
        'currencies': ['EUR', 'GBP', 'USD'],

        # transactions per hour of day
        'frac trans per hour': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'frac_trans_per_hour.npy')),
        # transactions per month in year
        'mean trans per month': np.load(join(utils_data.FOLDER_SIMULATOR_INPUT, 'mean_trans_per_month.npy')),

        # # countries
        # "country prob": pandas.read_csv('./data/aggregated/country_trans_prob.csv'),
        # # currencies per country
        # "currency prob per country": pandas.read_csv('./data/aggregated/currency_trans_prob_per_country.csv')

        # date

        # countries
    }

    return params


def get_path(parameters):
    """ given the parameters, get a unique path to store the outputs """
    # TODO
    path = './results/test'

    return path