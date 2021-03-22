#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from urllib.request import urlopen
from bs4 import BeautifulSoup


# In[3]:


url = "http://www.espn.com/nba/attendance/_/year/2019"
html = urlopen(url)


# In[4]:


soup = BeautifulSoup(html, 'lxml')
type(soup)


# In[5]:


title = soup.title
print(title)


# In[6]:


text = soup.get_text()


# In[7]:


soup.find_all('a')


# In[ ]:




