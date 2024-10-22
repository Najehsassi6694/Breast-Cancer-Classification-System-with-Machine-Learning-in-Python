#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


breast_cancer_dataset=sklearn.datasets.load_breast_cancer()


# In[4]:


print(breast_cancer_dataset)


# In[5]:


data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)


# In[6]:


data_frame.head()


# In[7]:


data_frame['label']=breast_cancer_dataset.target


# In[8]:


data_frame.tail(0)


# In[9]:


data_frame.shape


# In[10]:


data_frame.info()


# In[11]:


data_frame.isnull().sum()


# In[12]:


data_frame.describe()


# In[13]:


data_frame['label'].value_counts()


# In[14]:


data_frame.groupby('label').mean()


# In[15]:


X=data_frame.drop(columns='label',axis=1)
Y=data_frame['label']


# In[16]:


print(X)


# In[17]:


print(Y)


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[19]:


print(X.shape,X_train.shape,X_test.shape)


# In[20]:


model=LogisticRegression()


# In[21]:


model.fit(X_train,Y_train)


# In[23]:


X_train_prediction=model.predict(X_train)


# In[24]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)


# In[27]:


print('Accuracy on training data =',training_data_accuracy)


# In[28]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)


# In[29]:


print('Accuracy on test data =',test_data_accuracy)


# In[33]:


input_data=(18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0] == 0):
    print('the breast cancer is Malignant')
else:
        print('The breast cancer is Banign')

