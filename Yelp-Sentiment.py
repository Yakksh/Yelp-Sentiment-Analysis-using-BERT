#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings(action='ignore')


# ### Instantiate Model

# In[2]:


tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


# ### Encode and Calculate the Sentiment

# In[3]:


tokens = tokenizer.encode('this is okay but not great', return_tensors='pt')


# In[4]:


tokens


# In[5]:


tokenizer.decode(tokens[0])


# In[6]:


result = model(tokens)


# In[7]:


result


# In[8]:


result.logits


# In[9]:


int(torch.argmax(result.logits))+1


# ### Scrapping reviews from Yelp

# In[15]:


# agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'lxml')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]


# In[18]:


reviews[0]


# ### Scoring scrapped reviews

# In[19]:


def sentiment_score(review):
    tokens = tokenizer.encode('this is okay but not great', return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[20]:


import pandas as pd
import numpy as np


# In[21]:


data = pd.DataFrame(np.array(reviews), columns=['review'])


# In[24]:


data.head()


# In[29]:


data['sentiment']  = data['review'].apply(lambda x:sentiment_score(x))


# In[30]:


data

