#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[22]:


from matplotlib import pyplot as plt


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


url = "http://bit.ly/w-data"


# In[8]:


df= pd.read_csv(url)


# In[10]:


df.head()


# In[13]:


df.isnull().sum()


# In[15]:


df.describe()


# In[16]:


df.info()


# In[17]:


df.corr()


# In[18]:


plt.style.use('ggplot')


# In[24]:


df.plot(kind='line')
plt.title('hours vs percentage')
plt.xlabel('HOurs studied')
plt.ylabel('Percentage score')


# In[52]:


df.plot(kind='scatter', x='Hours', y='Scores',color='r',figsize=(8,5))
plt.title('Hours vs percentage')
plt.xlabel('hours')
plt.ylabel('percentage')
plt.show()


# In[31]:


df.plot(kind='scatter', x='Hours', y='Scores',color='r',figsize=(8,5))
plt.title('Hours vs percentage')
plt.xlabel('hours')
plt.ylabel('percentage')
plt.show()


# In[34]:


x=np.asanyarray(df[['Hours']])
y=np.asanyarray(df['Scores'])              
# using train test split to split data into train and test
train_x,test_x,train_y, test_y=train_test_split(x,y,test_size=0.2, random_state=2)
regressor= LinearRegression()
regressor.fit(train_x, train_y)


# In[36]:


print('Coefficient:', regressor.coef_)
print('Intercept:',regressor.intercept_)


# In[37]:


from sklearn import metrics


# In[38]:


from sklearn.metrics import r2_score


# In[39]:


y_pred=regressor.predict(test_x)


# In[40]:


print("r2_score: %.2f" % r2_score(y_pred,test_y))


# In[41]:


# comparing actual vs predicted
df2=pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})


# In[42]:


df2


# In[50]:


hours= 8
predicted_score=regressor.predict([[hours]])
print(f'No. of hours = {hours}')
print(f'Predicted score= {predicted_score[0]}')


# In[51]:





# In[ ]:




