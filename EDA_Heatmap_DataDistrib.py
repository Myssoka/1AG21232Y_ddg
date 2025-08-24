#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
df = pd.read_csv("Apr10FireProtDB_FeatGen_targetclass.csv")
df.head()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# df_corr = df.drop(['sequence',
#                    'protein_name',
#                    'uniprot_id',
#                    'secondary_structure',
#                   ], axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
# df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
# df_corr.drop(df_corr.iloc[1::2].index, inplace=True)
# df_corr_nd = df_corr.drop(df_corr[df_corr['Correlation Coefficient'] == 1.0].index)

# corr = df_corr_nd['Correlation Coefficient'] > 0.1
# df_corr_nd[corr]


# In[5]:


fig, ax = plt.subplots(nrows=1, figsize=(10, 5))  # ax an axs are different
sns.heatmap(df.drop(['sequence',
                     'protein_name',
                     'uniprot_id',
                     'secondary_structure',
                     'target_class',
                     'ddG',
                    ], axis=1).corr(), ax=ax, annot=True, square=True, cmap='coolwarm', annot_kws={'size': 8})
ax.set_title('Dataset Correlations', size=15)
plt.show()


# In[16]:


def pie_df(data):
    count = ""
    if isinstance(data, pd.DataFrame):
        count = data["target_class"].value_counts()
    else:
        count = data.value_counts()

    count.plot(kind = 'pie',
                figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
    plt.ylabel("Target Class: Neutral, Destabilizing, Stabilizing")
    plt.legend(["Destabilizing", "Neutral","Stabilizing"], loc= "upper right")
    plt.show()

pie_df(df)

