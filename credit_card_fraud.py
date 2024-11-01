from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')
from snapml import DecisionTreeClassifier


raw_data = pd.read_csv("C:/Users/byash/OneDrive/Desktop/IBM/course-9/creditcard.csv")
'''
print("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")
print(raw_data.head())
'''
labels = raw_data["Class"].unique()
size = raw_data["Class"].value_counts().values

fig,ax = plt.subplots(1,2,figsize=(10, 4), gridspec_kw={'wspace': 0.4})
ax[0].pie(size, labels = labels, autopct = '%1.3f%%' )
ax[0].set_title('Target Variable Value Counts')
#plt.show()

maxi = raw_data["Amount"].max()
mini = raw_data["Amount"].min()
q90 = raw_data["Amount"].quantile(0.9)

ax[1].hist(raw_data["Amount"].values, 6, histtype='bar', facecolor='g')
'''
plt.show()
print("Minimum amount value is ", np.min(raw_data.Amount.values))
print("Maximum amount value is ", np.max(raw_data.Amount.values))
print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))
'''

raw_data.iloc[:,1:30] = StandardScaler().fit_transform(raw_data.iloc[:,1:30])
data_matrix = raw_data.values
X = data_matrix[:,1:30]
y = data_matrix[:,30]
X = normalize(X,norm="l1")



X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.3, stratify=y)
#print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
#print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

w_train = compute_sample_weight('balanced', y_train)

from sklearn.tree import DecisionTreeClassifier

sklearn_dt = DecisionTreeClassifier(max_depth=4,random_state=35)
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))


snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time = time.time()-t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))
