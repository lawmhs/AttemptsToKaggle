# thanks to Wilson Shum from his post
# https://www.kaggle.com/wilsonshum/titanic-data-science-solutions/ 
# for his guided tutorial

# this is more or less a playground exploring his methodology

# for data anlysis
import pandas as pd
import random as rnd

# for visualization
import seaborn as sns
import matplotlib.pyplot as plt

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

# now here we are modifying the data, because some variables may not be relevant:

print('before', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
# this is just so we can see what things look like before we modify the data

# in pandas, DataFrame.drop means that we are dropping the labeled entry indicated 
# by the first variable (here we have an array of labels to drop) along the given axis
# 0 is dropping a row, 1 is dropping a column

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
# combine is an array of arrays, the 0th entry is the train_df, the 1st entry is test_df

print('after', train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
# as can be seen, two columns are lost from each of the objects

# this is the confusing part, using regex

for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

# so here we are creating a new feature by extracting the first word and a dot
# from the name string
# this gives us the title of the person

print(pd.crosstab(train_df['Title'], train_df['Sex']))

# according then to his analysis, we can see that some titles are more rare, such as 
# "Capt" or "Col" or "Countess"
# we can then replace them all with one category, "rare", to simplify things
# of course, survival rates may be weird for this one "rare" group
# but some groups only have one example so it would have been binary anyhow
# Bbut of course, think about the implications of such a thing on other datasets in the future


for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
	 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	# the above line replaces all occurences of rare titles with 'rare' and collects them under one category

	dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
	
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# now lets convert these titles to numbers so that they're easier to work with

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)
	
# presumably, fillna(0) means that we take any NA terms and make it 0
	
print(train_df.head())
# let's see what is happening

# great! Now we can drop the name and passenger ID since it doesn't matter too much to us now
print('Before : ', train_df.shape, test_df.shape)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1) # remember 1 denotes column
test_df = test_df.drop(['Name', 'PassengerId'], axis=1)
combine = [train_df, test_df]

print('After : ', train_df.shape, test_df.shape)


# now we convert M/F to 0/1

for dataset in combine:
	dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
	
# print(train_df.head)
	






