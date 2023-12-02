#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the provided dataset
file_path = 'train_normd.csv'
data = pd.read_csv(file_path)

data.head()


# In[2]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Separating the features and the target variable
X = data.drop('target', axis=1)
y = data['target']

# Performing PCA with 2 components
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with the two principal components as x and y, and the target variable as z
ax.scatter(X_pca_2[:, 0], X_pca_2[:, 1], y, c=y, cmap='viridis', marker='o')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Target Variable')

plt.show()


# In[3]:


# Performing PCA for the specified number of components
n_components = [10, 20, 30, 40]
explained_variances = {}

for n in n_components:
    pca = PCA(n_components=n)
    pca.fit(X)
    explained_variances[n] = sum(pca.explained_variance_ratio_)

explained_variances


# In[4]:


from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Function to perform k-fold cross-validation and return average score
def k_fold_cv_score(X, y, model, k=5):
    scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
    return np.mean(scores)

# Initialize RandomForestRegressor
rf_model = RandomForestRegressor()

# Number of components for PCA and the original dataset
n_components = [10, 20, 30, 40, X.shape[1]]  # Including original dataset's number of features

# Dictionary to store scores for each PCA version and the original dataset
scores = {}

# Performing k-fold cross-validation for each PCA version and the original dataset
for n in n_components:
    if n == X.shape[1]:  # Check for the original dataset
        scores['original'] = k_fold_cv_score(X, y, rf_model, k=5)
    else:
        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X)
        scores[f'pca_{n}'] = k_fold_cv_score(X_pca, y, rf_model, k=5)

scores
