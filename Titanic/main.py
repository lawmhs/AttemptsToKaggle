import pandas as pd
import random as rnd


import numpy as np
import scipy as sp
import sklearn as sk

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)

