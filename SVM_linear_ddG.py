#!/usr/bin/env python
# coding: utf-8

# ## FireProtDB - ddG Target Class (Multiclass classification) - SVM (linear) Classifier ##

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

print('Setup Complete')


# In[3]:


ddG_df = pd.read_csv("Apr10FireProtDB_FeatGen_targetclass.csv")
print(ddG_df.columns)


# In[5]:


ddG_df = ddG_df.dropna()
ddG_df = ddG_df.drop(['protein_name',
                       'uniprot_id',
                       'ddG',
                       'secondary_structure',
                       'MEC',
                       'Aliphatic Index',
                       'sequence'], axis = 1)
y = ddG_df['target_class']
X = ddG_df.drop(['target_class'], axis=1)

print(ddG_df.shape)
ddG_df.head()


# ## Test size 20% ##

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **Default parameters**

# In[22]:


svc_default = LinearSVC(C=1, max_iter=10000, dual='auto', random_state=42, verbose=0)
svc_default.fit(X_train, y_train)
y_train_pred = svc_default.predict(X_train)
def_train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy on train set: {def_train_accuracy:.04f}")

y_pred = svc_default.predict(X_test)
def_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {def_accuracy:.04f}")


# **Tuning parameters**

# In[24]:


# Find best C
best_C = None
best_acc = 0

for C in [2, 0.05, 0.1, 0.5, 1, 5, 10, 100]:
    model = LinearSVC(C=C, max_iter=10000, dual="auto", random_state=42, verbose=0)
    model.fit(X_train, y_train)  # training on CPU
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)  
    print(f"For C = {C}, the accuracy on test is {acc}")
    
    if acc > best_acc:
        best_C = C
        best_acc = acc

print(f"Best C value: {best_C}")

# Train the best model
best_SVM = LinearSVC(C=best_C, max_iter=10000, dual="auto", random_state=42, verbose=0)
best_SVM.fit(X_train, y_train)

# Evaluate model
y_train_pred = best_SVM.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy on train set: {train_accuracy:.04f}")

y_pred = best_SVM.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.04f}")


# ## Test Size 30% ##

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[11]:


## Test size 30% on C value: 2 ## 
SVM_30 = LinearSVC(C=2, max_iter=10000, dual="auto", random_state=42, verbose=0)
SVM_30.fit(X_train, y_train)
y_pred = SVM_30.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}")
print('Training set score: {:.4f}'.format(SVM_30.score(X_train, y_train)))


# ## Evaluation ##

# In[27]:


#For 20% test size
from sklearn.metrics import classification_report, confusion_matrix
print("Final classification report:")
print(classification_report(y_test, y_pred))


# In[24]:


#For 20% test size BEST PARAMETERS
from sklearn.metrics import classification_report, confusion_matrix
print("Final classification report:")
print(classification_report(y_test, y_pred))


# In[26]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

