#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE


# In[2]:


#attendance_data = pd.read_csv("Averages for MLB_NFL_NBA attendance_UPDATE - Sheet1.csv", header = None, keep_default_na = False, skiprows=1)
#attendance_data = attendance_data[:-2]
attendance_data = pd.read_csv("Averages for MLB_NFL_NBA attendance_UPDATE1 - Sheet1.csv") #header = None, keep_default_na = False, skiprows=1)
attendance_data = attendance_data[:-2]
#attendance_data = pd.read_csv("Averages for MLB_NFL_NBA attendance_UPDATE3 - Sheet1.csv", header = None, keep_default_na = False, skiprows=1)
#attendance_data = attendance_data[:-2]


# In[3]:


print(attendance_data)


# In[4]:


# could be useful if needed to convert strings in csv to integers
# data_numpy = df.to_numpy()
# df = df.replace(',','', regex = True)
# y = df[0]
# combined = df[4]
# mlb = df[1]
# nfl = df[2]
# nba = df[3]
# combined = df.select_dtypes(object).columns
# df[combined] = df[combined].apply(pd.to_numeric,errors='coerce')
# thanks emmanuel
# df.infer_objects()



# In[4]:


fig, ax = plt.subplots(figsize = (8,8)) #size of chart
#ax.plot(y, combined)
x1 = np.array([2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]) # year
y1 = np.array([1200723, 1239669, 1238897, 1243612, 1233412, 1231617, 1195648, 1180567, 318339 ]) # combined avg
y2mlb = np.array([2495309, 2467563, 2457987, 2458668, 2438636, 2422347, 2321658, 2282622, 0]) # mlb avg
y3nfl = np.array([536830, 540766, 544985, 541958, 529773, 539242, 534243, 527964, 62161]) # nfl avg
y4nba = np.array([570029, 710677, 713718, 730209, 731828, 733263, 731043, 731114, 574517]) # nba avg
plt.plot(x1, y1, marker = 'o')
plt.plot(x1, y2mlb, marker = 'o')
plt.plot(x1, y3nfl, marker = 'o')
plt.plot(x1, y4nba, marker = 'o')
plt.title("Attendance home game averages over 8 years")
plt.xlabel("Years")
plt.ylabel("Attendance")
plt.legend(["Combined Average", "MLB Average", "NFL Average", "NBA Average"])
plt.show()


# In[56]:


#convert values to integers since csv has strings
df = pd.DataFrame({'Year':[2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
                   'Combined Avg':[1200723, 1239669, 1238897, 1243612, 1233412, 1231617, 1195648, 1180567, 318339],
                   'MLB Avg':[2495309, 2467563, 2457987, 2458668, 2438636, 2422347, 2321658, 2282622, 0],
                   'NFL Avg':[536830, 540766, 544985, 541958, 529773, 539242, 534243, 527964, 62161],
                   'NBA Avg':[570029, 710677, 713718, 730209, 731828, 733263, 731043, 731114, 574517]})


# In[57]:


print(df) 


# In[58]:


# blue is good correlation and red is bad correlation
x = df['Year']
y = df['Combined Avg']
plt.figure(figsize=(8,8))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdYlBu)
plt.show


# In[59]:


target_corr = abs(cor["Combined Avg"]) #target variable
positive_corr_target = target_corr[target_corr >(0.5)] # anything less than point five is small correlation
positive_corr_target


# In[60]:


#Columns with the most correlation
print(df[["MLB Avg", "NFL Avg", "NBA Avg"]].corr())


# In[62]:


# drop unecceary columns if needed
df = df.drop(["Year"], axis=1)
df.head(n=5)


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.25) # test size is .25 and train size is .75
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape) 


# In[64]:


#setting up model
lrm = linear_model.LinearRegression()
lrm_model = lrm.fit(x_train, y_train)
lrm_predictions = lrm.predict(x_test)


# In[70]:


lrm_predictions[0:3]


# In[71]:


#plot of actual vs predicted 
plt.scatter(y_test, lrm_predictions)
plt.title("Actual vs Predicted")
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[72]:


print('Scores', lrm_model.score(x_test, y_test))


# In[73]:


print(lrm_predictions)
type(lrm_predictions)


# In[ ]:




