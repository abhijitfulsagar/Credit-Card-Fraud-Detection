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
outlierFraction = len(fraud) / len(valid)

print("Valid:{}".format(len(valid)))
print("Fraud:{}".format(len(fraud)))
#print("outlierFraction:{}".format(len(outlierFraction)))
print(outlierFraction)

# correlation matrix
corrmat = sampleDataset.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=0.8,square = True)
plt.show()

# get all the columns from sampleDataset
columns = sampleDataset.columns.tolist()

# filtering the columns to remove the the unnecessary data
columns = [c for c in columns if c not in ["Class"]]

X = sampleDataset[columns]
Y = sampleDataset["Class"]

# Anomaly detection

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# defining random_state
state = 0

# defining outlier detection methods
classifiers = {
               "Isolation Forest": IsolationForest(max_samples=len(X), 
                                                   contamination=outlierFraction,
                                                   random_state=0),
               "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                                    contamination=outlierFraction)                               
               }









