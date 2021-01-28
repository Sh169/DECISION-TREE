#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


fraud = pd.read_csv("Fraud_check.csv")
data = fraud.head()


# In[4]:


fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"


# In[5]:


fraud["income"].unique()


# In[6]:


fraud["income"].value_counts()


# In[15]:


### dropping the Taxable.Income columns


# In[16]:


fraud.rename(columns={"Marital.Status":"marital","City.Population":"population","Work.Experience":"workexp"},inplace=True)


# In[17]:


fraud.head()


# In[18]:


fraud.isnull().sum()


# In[20]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass


# In[21]:


features = fraud.iloc[:,0:5]
labels = pd.DataFrame(fraud.iloc[:,5])
fraud["income"].value_counts()


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.3, stratify=labels)


# In[23]:


print(y_train["income"].value_counts())
print(y_test["income"].value_counts())


# In[24]:


##Converting the column names into the list format
colnames = list(fraud.columns)
predictors = colnames[:5]
target = colnames[5]

fraud.info()


# In[25]:


from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(x_train,y_train)


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


##Prediciton on train data 
pred_train = pd.DataFrame(model.predict(x_train))


# In[28]:


### Finding the accuracy of train data
acc_train = accuracy_score(y_train,pred_train) #100%


# In[29]:


##Confusion matrix for train data
from sklearn.metrics import confusion_matrix


# In[30]:


cm = pd.DataFrame(confusion_matrix(y_train,pred_train))


# In[31]:


##Prediction on test data
pred_test = pd.DataFrame(model.predict(x_test))


# In[32]:


acc_test = accuracy_score(y_test,pred_test)


# In[33]:


#confusion matrix for test data
cm_test = confusion_matrix(y_test,pred_test)


# In[35]:


pip install six


# In[36]:


pip install graphviz


# In[37]:


pip install pydotplus


# In[39]:


import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib


# In[63]:


import warnings
warnings.filterwarnings('ignore')


# In[75]:


##Visualizing the decision trees
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()


# In[82]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=predictors.feature_names,  
                   class_names=fraud.target_names,
                   filled=True)


# In[ ]:




