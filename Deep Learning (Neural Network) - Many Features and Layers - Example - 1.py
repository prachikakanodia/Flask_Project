#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv('pima-indians-diabetes.data.csv', header=None)


# In[3]:


dataset.info()


# In[4]:


dataset


# In[5]:


dataset.head(2)


# In[6]:


y = dataset[8]


# In[7]:


dataset.columns


# In[8]:


X = dataset[[0,1,2,3,4,5,6,7]]


# In[9]:


from keras.models import Sequential


# In[10]:


from keras.layers import Dense


# In[11]:


from keras.optimizers import Adam


# In[12]:


model = Sequential()


# In[13]:


model.add(Dense(units=10, input_dim=8, activation='relu'))


# In[14]:


model.add(Dense(units=8, activation='relu'))


# In[15]:


model.add(Dense(units=1, activation='sigmoid'))


# In[16]:


model.summary()


# In[17]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[18]:


model.fit(X, y, epochs=100)


# In[19]:


model.save("dia_model.h5")


# In[ ]:




