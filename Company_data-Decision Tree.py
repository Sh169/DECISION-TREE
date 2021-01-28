#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as pltfrom 
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier


# In[2]:


Company= pd.read_csv("Company_Data.csv")
Company.head()


# In[4]:


#to check if we have null valuesin dataset or not
Company.isnull().any()


# In[8]:


#To check data type
Company.dtypes


# In[9]:


# to check the summary of the dataset
Company.describe()


# In[11]:


Company.info()


# In[14]:


Company['ShelveLoc'],class_names = pd.factorize(Company['ShelveLoc'])
Company.ShelveLoc
print(class_names)


# In[15]:


Company['Urban'],class_names2=pd.factorize(Company['Urban'])
print(class_names2)


# In[16]:


Company['US'].replace(to_replace=['Yes', 'No'],value= ['0', '1'], inplace=True)
Company.US


# In[17]:


print(Company['US'].unique())
Company.info()


# In[18]:


Company['Sales'].plot.hist()
plt.show()


# In[19]:


Company.Sales.describe()


# In[20]:


#Converting the Sales column which is continuous into categorical
category = pd.cut(Company.Sales,bins=[0,5.39,9.32,17],labels=['low','moderate','high'])
Company.insert(0,'Sales_Group',category)


# In[21]:


Company.drop(['Sales'],axis = 1, inplace = True)


# In[22]:


import seaborn as sns
sns.pairplot(Company)


# In[23]:


Company['Sales_Group'].unique()
Company.Sales_Group.value_counts()


# #### Feature Selection

# In[26]:


colnames=list(Company.columns)
colnames
predictors=colnames[1:]
predictors 


# In[29]:


target=colnames[0]
target


# ###### Splitting data into training and testing data set

# In[64]:


import numpy as np

#np.random.uniform(start,stop,size) will generate array of real numbers with size = size
Company['is_train'] = np.random.uniform(0, 1, len(Company))<= 0.75
Company['is_train']
train,test = Company[Company['is_train'] == True],Company[Company['is_train']==False]


# In[66]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(Company,test_size = 0.2)


# In[1]:


from sklearn.tree import  DecisionTreeClassifier


# In[68]:


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])


# In[69]:


preds = model.predict(test[predictors])
pd.Series(preds).value_counts()


# In[70]:


pd.crosstab(test[target],preds)


# In[74]:


# Accuracy = train
np.mean(train.Sales_Group == model.predict(train[predictors]))


# In[75]:


# Accuracy = Test
np.mean(preds==test.Sales_Group)


# ### Plot Decision Tree
# 

# In[81]:


pip install --upgrade scikit-learn==0.23.1


# In[91]:


pip install six


# In[103]:


pip install graphviz


# In[105]:


pip install pydotplus


# In[124]:


import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib


# In[127]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()


# In[136]:


pip install pydot


# In[153]:


export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = predictors,class_names=['low','moderate','high'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  


# In[155]:


#graph.write_png('Sales.png')
#Image(graph.create_png())


# In[160]:





# In[ ]:




