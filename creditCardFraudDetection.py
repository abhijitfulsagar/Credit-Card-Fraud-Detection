# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:36:21 2018

@author: ABHIJIT
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import seaborn as sns
import sklearn

# getting the dataset
dataset = pd.read_csv("creditcard.csv")

# exploring the dataset
print(dataset.columns)
print(dataset.shape)
print(dataset.describe())

sampleDataset = dataset.sample(frac = 0.2,random_state=0)
print(sampleDataset.shape)

# plotting the histograms
sampleDataset.hist(figsize = (20,20))
plt.show()

# determinig the number of fraud cases in the sample dataset
fraud = sampleDataset[sampleDataset['Class'] == 1]
valid = sampleDataset[sampleDataset['Class'] == 0]
outlierFraction = len(fraud) / float(len(valid))

print("Valid:{}".format(len(valid)))
print("Fraud:{}".format(len(fraud)))
print("outlierFraction:{}".format(len(outlierFraction)))















