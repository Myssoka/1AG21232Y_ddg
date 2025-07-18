#!/usr/bin/env python
# coding: utf-8

# ## FireProtDB - ddG Target Class (Multiclass classification) - Softmax Regression Classifier ##

# In[1]:


#Initial setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression

import warnings 
warnings.filterwarnings("ignore")

print("Setup Complete")


# In[2]:


ddG_df = pd.read_csv('Apr10FireProtDB_FeatGen_targetclass.csv')
print(ddG_df.columns)
ddG_df.head()


# In[5]:


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


# In[6]:


print(ddG_df.shape)
ddG_df.head()


# ## MinMaxScaler ##

# In[14]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_train = pd.DataFrame(X_train, columns = [cols])
# X_test = pd.DataFrame(X_test, columns = [cols])


# ## Test size 20% ##

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

cols = X_train.columns

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

#Testing set accuracy
y_pred_test = logreg.predict(X_test)
print('Model (test set) accuracy score: {0:04f}'.format(accuracy_score(y_test, y_pred_test)))
#Training set accuracy
y_pred_train = logreg.predict(X_train)
print('Model (train set) accuracy score: {0:04f}'.format(accuracy_score(y_train, y_pred_train)))


# ## Test size 30% ##

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

cols = X_train.columns

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

#Testing set accuracy
y_pred_test = logreg.predict(X_test)
print('Model (test set) accuracy score: {0:04f}'.format(accuracy_score(y_test, y_pred_test)))
#Training set accuracy
y_pred_train = logreg.predict(X_train)
print('Model (train set) accuracy score: {0:04f}'.format(accuracy_score(y_train, y_pred_train)))


# ## GridSearch for best parameters ##

# In[14]:


param_grid_logreg = {
    'penalty': ['l2'],  # Regularization method (L2 is used for softmax)
    'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength (smaller = stronger regularization)
    'solver': ['lbfgs', 'saga'],  # Optimization algorithm (saga supports large datasets and multiclass)
    'multi_class': ['multinomial'],  # Use softmax (multinomial) for multiclass problems
    'max_iter': [1000]  # Maximum number of iterations for convergence
}


# In[22]:


grid_search_cv = GridSearchCV(estimator=logreg, param_grid=param_grid_logreg, cv=5, n_jobs=1, verbose=0, scoring='accuracy')

grid_search_cv.fit(X_train, y_train)

best_params = grid_search_cv.best_params_
best_model = grid_search_cv.best_estimator_

print('Best Parameters: ', best_params)
print('Best Model: ', best_model)

best_accuracy = grid_search_cv.best_score_
print(f'Best Cross-Validation Accuracy: {best_accuracy:.2f}')


# ## Best parameters ##

# In[30]:


logreg_best = LogisticRegression(C=0.01, max_iter=1000, multi_class='multinomial', random_state=0)
logreg_best.fit(X_train, y_train)
y_pred_best = logreg_best.predict(X_test)
print("Softmax regression best accuracy (best parameters): {0:04f}".format(accuracy_score(y_test, y_pred_best)))
print('Training set score: {:.4f}'.format(logreg_best.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(logreg_best.score(X_test, y_test)))


# ## Evaluation ##

# In[21]:


#For 20% test size
from sklearn.metrics import classification_report, confusion_matrix
print("Final classification report:")
print(classification_report(y_test, y_pred_test))


# In[32]:


#For 20% test size BEST PARAMETERS
from sklearn.metrics import classification_report, confusion_matrix
print("Final classification report:")
print(classification_report(y_test, y_pred_test))


# In[23]:


sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

