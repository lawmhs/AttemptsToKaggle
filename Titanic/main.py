# thanks to Wilson Shum from his post
# https://www.kaggle.com/wilsonshum/titanic-data-science-solutions/ 
# for his guided tutorial

# this is more or less a playground exploring his methodology

# for data anlysis
import pandas as pd
import random as rnd

# for visualization
import seaborn as sns
import matplotlib.pyplot

# MACHINE LEARNING STUFF
import numpy as np
import scipy as sp
import sklearn as sk

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)

# for now lets work with Passenger ID, Pclass, Sex, Age

# the following prints sample statistics for each of the features
# print(train_df.describe())

# of course, something like "mean" is not too helpful for features like
# the ticket class or whatnot, it just tells us that there were a lot of third class passengers

# pivoting on features example from tutorial:

# pivoting on class to see survival rates of each class, as one can note,
# the richer you were (equivalent, the better your ticket class), the more likely you survived
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# pivoting on sex: we can guess that because of the "women and children" policy,
# we will see a positive correlation with female and surviving, and a negative one for male and surviving
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
# will output a table with columns "sex" and "survived"
# the pivoting

# data visualization

g = sns.FacetGrid(train_def, col='Survived')
g.map(plt.hst, 'Age', bins=20)

