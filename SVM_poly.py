#!/usr/bin/env python
# coding: utf-8

# ## FireProtDB - ddG Target Class (Multiclass classification) - SVM Poly kernel ##

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score


print('Setup Complete')


# In[8]:


ddG_df = pd.read_csv('Apr10FireProtDB_FeatGen_targetclass.csv')
ddG_df = ddG_df.dropna()
ddG_df = ddG_df.drop(['protein_name',
                      'uniprot_id',
                      'ddG',
                      'secondary_structure',
                      'sequence',
                      'MEC',
                      'Aliphatic Index'], axis=1)

y = ddG_df['target_class']
X = ddG_df.drop(['target_class'], axis=1)


# ## Polynomial (poly) ##

# **Test size 20%**

# In[20]:


from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# **Default parameters**

# In[23]:


poly_default = SVC(kernel= 'poly', C=1)
poly_default.fit(X_train, y_train)

y_train_default_pred = poly_default.predict(X_train)
default_train_accuracy = accuracy_score(y_train, y_train_default_pred)
print(f"Accuracy on train set: {default_train_accuracy:.04f}")

y_pred = poly_default.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.04f}")


# In[27]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import accuracy_score\n\nparam_grid = {\'C\': [2, 0.05, 0.1, 0.5, 1, 5, 10, 100]}\n\nsvm_model = SVC(kernel = \'poly\')\n\ngrid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring=\'accuracy\')\ngrid_search.fit(X_train, y_train)\n\n# Print accuracy for each C\nfor mean_score, C_value in zip(grid_search.cv_results_[\'mean_test_score\'], param_grid[\'C\']):\n    print(f"For C = {C_value}, the cross-validated accuracy is {mean_score:.4f}")\n\nbest_C = grid_search.best_params_[\'C\']\nprint(f"Best C value: {best_C}")\n\nbest_SVM = SVC(kernel= \'poly\', C=best_C)\nbest_SVM.fit(X_train, y_train)\n\ny_train_pred = best_SVM.predict(X_train)\ntrain_accuracy = accuracy_score(y_train, y_train_pred)\n\nprint(f"Accuracy on train set: {train_accuracy:.04f}")\n\ny_pred = best_SVM.predict(X_test)\naccuracy = accuracy_score(y_test, y_pred)\n\nprint(f"Accuracy on test set: {accuracy:.04f}")\n')


# **Test size 30%, C=1**

# In[16]:


from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[18]:


best_SVM = SVC(kernel= 'poly', C=1)
best_SVM.fit(X_train, y_train)

#Train accuracy
y_train_pred = best_SVM.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Accuracy on train set: {train_accuracy:.4f}")

#Test accuracy
y_pred = best_SVM.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}")


# ## Evaluation - poly ##

# In[25]:


#For 20% test size
from sklearn.metrics import classification_report, confusion_matrix
print("Final classification report:")
print(classification_report(y_test, y_pred))


# In[29]:


#For 20% test size BEST PARAMETERS
from sklearn.metrics import classification_report, confusion_matrix
print("Final classification report:")
print(classification_report(y_test, y_pred))


# In[47]:


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

