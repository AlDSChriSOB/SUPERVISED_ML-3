#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
# seaborn and matplotlib makes plots
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df = pd.read_csv("train-en.csv")
train_df


# In[3]:


df_eval = pd.read_csv("eval-en.csv")
df_eval


# In[4]:


challenge_df = pd.read_csv("challenge-en.csv")
challenge_df


# In[5]:


train_df.describe(include = 'all')


# In[6]:


train_df.head()


# In[7]:


#  list columns as some are truncated from view
for column in train_df.columns:
    print(column)


# In[8]:


train_df.info()


# In[9]:


train_df.groupby('wind_speed48M')['Output'].describe()


# In[10]:


# Training data.

x_train= pd.read_csv('train-en.csv', usecols=["wind_speed48M"])
y_train= pd.read_csv('train-en.csv', usecols=["Output"])
y_test = pd.read_csv('eval-en.csv', usecols=["Output"])
x_test = pd.read_csv('eval-en.csv', usecols=["wind_speed48M"])


# In[11]:


plt.scatter(x_train["wind_speed48M"], y_train["Output"], color="cyan")
plt.show()


# In[12]:


#sns.pairplot(df_eval)


# In[20]:


reg = LinearRegression().fit(x_train, y_train)
scores = reg.score(x_train, y_train)


# In[14]:


scores


# In[15]:


x_test = pd.read_csv('eval-en.csv', usecols=["wind_speed48M"])


# In[16]:


predictions = reg.predict(x_test)
print(predictions)


# In[17]:


plt.scatter(x_test["wind_speed48M"], y_test["Output"], color="cyan")
plt.plot(x_test, predictions, color="red", linewidth=3)
plt.show()


# In[18]:


train_df.corr()


# In[19]:


train_df[['wind_speed48M','Output']].corr()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




