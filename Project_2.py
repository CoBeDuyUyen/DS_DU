#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction:
# Nowadays, exployers are concerned about the factors that their employees quit job or their job satisfaction. We have a dataset created by IBM data scientists. This report will help us answer three questions:
# 1. What are key factors that are playing into current attrition rates?
# 2. What are key factors that are playing into current satisfaction rates?
# 3. When are employees leaving?
# 
# Firstly, we will import some libraries supporting the code.

# In[234]:


import pandas as pd
import numpy as np
import scipy.stats as sp
import math
from scipy.stats import t
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns


# # 2. Describe the data:
# 

# In[235]:


data = pd.read_csv("F:\PJ2.csv")
data.head(10)


# In[236]:


data.describe()


# # 3. Vizualization:
# 1. Key factors that are playing into current attrition rates:
# We will convert the data type of 'Attrition' column to numeric.'Yes' is 1 and 'No' is 0.

# In[237]:


def segment1(x): #Convert data function
    if x == 'Yes': return 1
    else: return 0
data['Attrition'] = data.Attrition.apply(lambda x: segment1(x)) #Convert data 


# Then, we continue to classify numeric columns to 2 types:
# - Column has over 30 values
# - Column has less than 30 values
# We will vizualize correlation of 'Attrition' and other numeric features.

# In[238]:


cont_col = []
for column in data.columns:
    if data[column].dtypes != object and data[column].nunique() > 30:
        cont_col.append(column)
plt.figure(figsize=(20, 10))

for i, column in enumerate(cont_col, 1):
    plt.subplot(2, 4, i)
    data[data["Attrition"] == 0][column].hist(bins=35, color='blue', label='Attrition = NO', alpha=0.6)
    data[data["Attrition"] == 1][column].hist(bins=35, color='red', label='Attrition = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[258]:


disc_col = []
for column in data.columns:
    if data[column].dtypes != object and data[column].nunique() < 30:
        disc_col.append(column)
disc_col.remove('Attrition')
plt.figure(figsize=(20, 20))

for i, column in enumerate(disc_col, 1):
    plt.subplot(5, 4, i)
    data[data["Attrition"] == 0][column].hist(bins=35, color='blue', label='Attrition = NO', alpha=0.6)
    data[data["Attrition"] == 1][column].hist(bins=35, color='red', label='Attrition = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# From these charts, we can answer the third question: When are employees leaving?
# Employees may quit job within 10 years at company. They often work from 0 to 2.5 years in current role and from 2.5 to 10 years since their last promotion.

# We consider character type features:
# 

# In[240]:


BT = px.bar(data, x="BusinessTravel", facet_col="Attrition", color_discrete_sequence = ["red"])
BT.show()


# In[241]:


department = px.bar(data, x="Department", facet_col="Attrition", color_discrete_sequence = ["red"])
department.show()


# In[ ]:


The chart shows us that the ratio employees who belong to Sales may quit job more than others.


# In[242]:


educationfield = px.bar(data, x="EducationField", facet_col="Attrition", color_discrete_sequence = ["red"])
educationfield.show()


# In[243]:


Gender1 = px.pie(data, names='Gender', title = 'Gender of employees', color_discrete_sequence = px.colors.sequential.RdBu)
Gender1.show()
Gender2 = px.pie(data[data.Attrition == 1], names='Gender', title = 'Gender of employees who want to quit job', color_discrete_sequence = px.colors.sequential.RdBu)
Gender2.show()


# In[244]:


JR = px.bar(data, x="JobRole", facet_col="Attrition", color_discrete_sequence = ["red"])
JR.show()


# In[245]:


MS = px.bar(data, x="MaritalStatus", facet_col="Attrition", color_discrete_sequence = ["red"])
MS.show()


# 

# In[246]:


Overtime1 = px.pie(data, names='OverTime', title = 'Over time rate of employees', color_discrete_sequence = px.colors.sequential.RdBu)
Overtime1.show()
Overtime2 = px.pie(data[data.Attrition == 1], names='OverTime', title = 'Over time rate of employees who want to quit job', color_discrete_sequence = px.colors.sequential.RdBu)
Overtime2.show()


# Conclusions:
# We can see that some features are concerned with 'Attrition' like Age, MonthlyIncome, YearAtCompany, TotalWorkingYears, YearsAtCompany, TrainingTimesLastYear, WorkLifeBalance, MaritalStatus, OverTime.

# 2. Key factors that are playing into current satisfaction rates:
# 

# In[256]:


cont_col = []
for column in data.columns:
    if data[column].dtypes != object:
        cont_col.append(column)
cont_col.remove("Attrition")
cont_col.remove("JobSatisfaction")
for i, column in enumerate(cont_col, 0):
    fig = px.box(data, x="JobSatisfaction", y=cont_col[i])
    fig.show()


# We can see that only HourlyRate and EmployeeNumber really concerned with JobSatisfaction. If these numbers increase, the job satisfaction rate reduce. 

# We consider character type features:

# In[248]:


a=["1", "2", "3", "4"]
dt1 = data[data.JobSatisfaction == 1]
dt2 = data[data.JobSatisfaction == 2]
dt3 = data[data.JobSatisfaction == 3]
dt4 = data[data.JobSatisfaction == 4]
yes = [len(dt1[dt1.Attrition == 0]), len(dt2[dt2.Attrition == 0]),
        len(dt3[dt3.Attrition == 0]), len(dt4[dt4.Attrition == 0])]
no = [len(dt1[dt1.Attrition == 1]), len(dt2[dt2.Attrition == 1]),
        len(dt3[dt3.Attrition == 1]), len(dt4[dt4.Attrition == 1])]
fig = go.Figure(data=[
    go.Bar(name='YES', x=a, y=yes),
    go.Bar(name='NO', x=a, y=no)
])
fig.update_layout(barmode='stack')
fig.show()


# The chart shows us that if someone want to quit his job, he may not satisfied with his job.

# In[249]:


Non_Travel = [len(dt1[dt1.BusinessTravel == "Non-Travel"]), len(dt2[dt2.BusinessTravel == "Non-Travel"]),
        len(dt3[dt3.BusinessTravel == "Non-Travel"]), len(dt4[dt4.BusinessTravel == "Non-Travel"])]
Travel_Rarely = [len(dt1[dt1.BusinessTravel == "Travel_Rarely"]), len(dt2[dt2.BusinessTravel == "Travel_Rarely"]),
        len(dt3[dt3.BusinessTravel == "Travel_Rarely"]), len(dt4[dt4.BusinessTravel == "Travel_Rarely"])]
Travel_Frequently = [len(dt1[dt1.BusinessTravel == "Travel_Frequently"]), len(dt2[dt2.BusinessTravel == "Travel_Frequently"]),
        len(dt3[dt3.BusinessTravel == "Travel_Frequently"]), len(dt4[dt4.BusinessTravel == "Travel_Frequently"])]
fig = go.Figure(data=[
    go.Bar(name='Non_Travel', x=a, y=Non_Travel),
    go.Bar(name='Travel_Rarely', x=a, y=Travel_Rarely),
    go.Bar(name='Travel_Frequently', x=a, y=Travel_Frequently)
])
fig.update_layout(barmode='stack')
fig.show()


# BusinessTravel may not affect to JobSatisfaction. The ratio of 3 parts doesn't change too much.

# In[250]:


de = px.bar(data, x="Department", facet_col="JobSatisfaction", color_discrete_sequence = ["red"])
de.show()


# In[254]:


data[data["Gender"] == "Male"]["JobSatisfaction"].hist(bins=35, color='blue', label='Male', alpha=0.6)
data[data["Gender"] == "Female"]["JobSatisfaction"].hist(bins=35, color='red', label='Female', alpha=0.6)
plt.legend()
plt.xlabel('Gender')


# We can see that men easy to feel satisfied with his job than women.

# In[257]:


data[data["OverTime"] == "Yes"]["JobSatisfaction"].hist(bins=35, color='blue', label='Yes', alpha=0.6)
data[data["OverTime"] == "No"]["JobSatisfaction"].hist(bins=35, color='red', label='No', alpha=0.6)
plt.legend()
plt.xlabel('OverTime')


# The chart shows us that if somebody doesn't work over time, he may satisfied more with his job.

# Conclusions:
# We can see that some features are concerned with 'JobSatisfaction' like Attrition, HourlyRate, EmployeeNumber, Gender, OverTime.

# # 4. Conclusion:
# - There are many reasons that make employees want to quit job like: Age, MonthlyIncome, YearAtCompany, TotalWorkingYears, YearsAtCompany,...
# - There are also many factors which play into satisfaction rate like: Attrition, HourlyRate, EmployeeNumber, Gender, OverTime.
# - Employees may quit job when they have worked for 0 to 2.5 years in current role and from 2.5 to 10 years since their last promotion. They rarely work for a company over 10 years.
