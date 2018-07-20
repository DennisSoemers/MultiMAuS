"""
Script to generate a csv file with data (input parameters) for running multiple experiments in parallel using
the Worker framework (https://www.vscentrum.be/cluster-doc/running-jobs/worker-framework)

@author Dennis Soemers
"""

import os
import pandas as pd
import random

NUM_EXPERIMENTS = 25

if __name__ == '__main__':
    df = pd.DataFrame(columns=['seed', 'run_idx'])
    df['seed'] = [random.randrange(2**31) for _ in range(NUM_EXPERIMENTS)]
    df['run_idx'] = range(NUM_EXPERIMENTS)

    df.to_csv(os.path.join(os.path.dirname(__file__), 'workers_data.csv'), index=False)
