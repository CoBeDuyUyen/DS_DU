#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as sp
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from statsmodels.formula.api import ols
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode


# # 2. Exploring the data:

# In[2]:


data = pd.read_csv("F:\PJ3.csv")
data.head(10)


# In[3]:


data.shape


# In[4]:


display(data.isnull().sum())


# In[5]:


data.drop(['BuildingArea'], inplace = True, axis = 1)
data.drop(['YearBuilt'], inplace = True, axis = 1)
data


# In[6]:


for column in data.columns:
    if data[column].nunique() == 1:
        data.drop(column, inplace = True, axis = 1)
data.drop('Address', inplace = True, axis = 1)
data


# In[7]:


for column in data.columns:
    if data[column].dtypes != object:
        data[column].replace(np.NaN, np.mean(data[column]), inplace=True)
    else:
        data[column].replace(np.NaN, "non-value", inplace=True)
data


# In[8]:


data.describe(percentiles = [.1, .25, .5, .75, .97, .999])


# In[9]:


data = data[data.Rooms <= 7]
data = data[data.Price <= 6000000]
data = data[data.Bedroom2 <= 7]
data = data[data.Bathroom <= 5]
data = data[data.Car <= 8]
data = data[data.Landsize <= 20000]
data


# # 3. Vizualization:

# In[10]:


for column in data.columns:
    if data[column].dtypes != object:
        fig = px.histogram(data, x=column, nbins=20)
        fig.show()


# In[11]:


P = [np.mean(data[data.Rooms == 0].Price), np.mean(data[data.Rooms == 1].Price), np.mean(data[data.Rooms == 2].Price), np.mean(data[data.Rooms == 3].Price), np.mean(data[data.Rooms == 4].Price), 
     np.mean(data[data.Rooms == 5].Price), np.mean(data[data.Rooms == 6].Price), np.mean(data[data.Rooms == 7].Price)]
fig = px.line(x = [0, 1, 2, 3, 4, 5, 6, 7], y = P,
             labels=dict(x="Rooms", y="Price", color="Time Period"))
fig.show()


# In[12]:


P = [np.mean(data[data.Bedroom2 == 0].Price), np.mean(data[data.Bedroom2 == 1].Price), np.mean(data[data.Bedroom2 == 2].Price), np.mean(data[data.Bedroom2 == 3].Price), 
     np.mean(data[data.Bedroom2 == 4].Price), np.mean(data[data.Bedroom2 == 5].Price), np.mean(data[data.Bedroom2 == 6].Price), np.mean(data[data.Bedroom2 == 7].Price)]
fig = px.line(x = [0, 1, 2, 3, 4, 5, 6, 7], y = P,
             labels=dict(x="Bedroom2", y="Price", color="Time Period"))
fig.show()


# In[13]:


P = [np.mean(data[data.Bathroom == 0].Price), np.mean(data[data.Bathroom == 1].Price), np.mean(data[data.Bathroom == 2].Price), 
     np.mean(data[data.Bathroom == 3].Price),  np.mean(data[data.Bathroom == 4].Price), np.mean(data[data.Bathroom == 5].Price)]
fig = px.line(x = [0, 1, 2, 3, 4, 5], y = P,
             labels=dict(x="Bathroom", y="Price", color="Time Period"))
fig.show()


# In[14]:


P = [np.mean(data[data.Car == 0].Price), np.mean(data[data.Car == 1].Price), np.mean(data[data.Car == 2].Price), np.mean(data[data.Car == 3].Price), 
     np.mean(data[data.Car == 4].Price), np.mean(data[data.Car == 5].Price), np.mean(data[data.Car == 6].Price), np.mean(data[data.Car == 7].Price),
     np.mean(data[data.Car == 8].Price)]
fig = px.line(x = [0, 1, 2, 3, 4, 5, 6, 7, 8], y = P,
             labels=dict(x="Bedroom2", y="Price", color="Time Period"))
fig.show()


# In[15]:


fig = px.scatter(data, x="Price", y="Distance")
fig.show()


# In[16]:


fig = px.scatter(data, x="Price", y="Landsize")
fig.show()


# In[17]:


fig = px.scatter(data, x="Price", y="Propertycount")
fig.show()


# In[18]:


def formatting_columns(x):
    x[0] = '{0:.2f}'.format(x[0])
    x[1] = '{0:.0f}'.format(x[1])
    x[2] = '{0:.0f}'.format(x[2])
    x[3] = '{0:.0f}'.format(x[3])
    return x
data['Price'].groupby(data.Suburb).agg(['mean','count','max','min']).sort_values(by='mean',ascending=False).apply(formatting_columns,axis=1).head(20)


# In[19]:


iris_lm=ols('Price ~ Suburb + Rooms + Type + Method + SellerG + Distance + Postcode + Bedroom2 + Bathroom + Car + Landsize + CouncilArea + Lattitude + Longtitude + Regionname + Propertycount', data=data).fit() #Specify C for Categorical
sm.stats.anova_lm(iris_lm, typ=2)


# # 4. Linear regression:

# In[49]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
LR = LinearRegression() 
all_Data = []
suburb_dummies = pd.get_dummies(data[["Type", "Method"]])
all_Data = data.drop(["Price", "Suburb","Date", "SellerG", "Type", "Method", "CouncilArea", "Regionname"],axis=1).join(suburb_dummies)
X = all_Data
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
LR.fit(X_train,y_train)
predictions = LR.predict(X_test)
plt.scatter(y_test, predictions)


# # 5. Holdout and cross validation:

# In[50]:


from sklearn.metrics import mean_squared_error
all_Data = []
suburb_dummies = pd.get_dummies(data[["Type", "Method"]])
all_Data = data[['Rooms','Distance','Car','Bathroom']].join(suburb_dummies)
X = all_Data
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
LR.fit(X_train,y_train)
y_predict = LR.predict(X_test)
print(y_test)
print("Mean squared error is: %.4f"%mean_squared_error(y_test, y_predict))


# In[51]:


print(y_predict)


# In[52]:


predicted = cross_val_predict(LR, X.values, y.values, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# # 6. Residual plot:

# In[53]:


d1 = pd.DataFrame()
d1['test'] = y_test
d1['predict'] = y_predict
d1.sort_values(by=['test'], inplace=True)
dy = d1['test'] - d1['predict']
fig, ax = plt.subplots()
ax.plot(d1['test'],d1['predict'])
ax.scatter(d1['test'],d1['predict']+dy)

ax.vlines(d1['test'],d1['predict'],d1['predict']+dy)

plt.show()


# In[ ]:





# In[ ]:




