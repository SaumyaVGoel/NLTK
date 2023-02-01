#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install nltk


# In[5]:


import nltk


# In[6]:


nltk.download('all')


# In[7]:


import nltk
import matplotlib as plt
import pandas as pd
import math
from nltk.corpus import brown
import numpy as np
nltk.download('brown')
from nltk.corpus import brown
brown.categories()


# In[8]:


print("Length of words is: ")
len(brown.words()) #tokens


# In[13]:


number_of_words = 0
with open('nlp.txt','r') as file:
	data = file.read()
	lines = data.split()
	number_of_words += len(lines)
print("Total number of words:")
print(number_of_words)


# In[14]:


f = open("nlp.txt", 'r', errors = 'ignore')
x = f.read()
x


# In[15]:


from collections import Counter
the_count = Counter(x)
print(the_count)
     


# In[16]:


brown.words(categories='government')


# In[17]:


from collections import Counter
split_it = x.split()
Counter = Counter(split_it)
most_occur = Counter.most_common(4)
print(most_occur)


# In[18]:


nltk.download('punkt') 
nltk.download('wordnet') 
from nltk.tokenize import sent_tokenize, word_tokenize
sentences= nltk.sent_tokenize(x)
length= len(sentences)
length


# In[19]:


nltk.download('treebank')
from nltk.corpus import treebank
print(treebank.fileids())
print(treebank.words('wsj_0003.mrg'))
print(treebank.tagged_words('wsj_0003.mrg'))
print(treebank.parsed_sents('wsj_0003.mrg')[0])


# In[20]:


#POS tagged corpora
from nltk.corpus import brown
print(brown.words())
print(brown.tagged_words())
print(brown.sents())
print(brown.tagged_sents())
print(brown.paras(categories='reviews'))
print(brown.tagged_paras(categories='reviews'))


# In[21]:


nltk.download('indian')
from nltk.corpus import indian
print(indian.words()) 
print(indian.tagged_words()) 
     


# In[22]:


nltk.download('universal_tagset')
nltk.download('conll2000')
nltk.download('switchboard')
print(brown.tagged_sents(tagset='universal'))
from nltk.corpus import conll2000, switchboard
print(conll2000.tagged_words(tagset='universal'))


# In[23]:



#3
x = x.lower()
     

nltk.download('punkt') 
nltk.download('wordnet') 
from nltk.tokenize import sent_tokenize, word_tokenize
sent_tokens = nltk.sent_tokenize(x) 
word_tokens = nltk.word_tokenize(x) 
sent_tokens


# In[24]:


word_tokens


# In[25]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
tokens = word_tokenize(x)
filtered_text = [t for t in tokens if not t in stopwords.words("english")]
print(" ".join(filtered_text))


# In[26]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd 
ps = PorterStemmer()
tokens = word_tokenize(x)
print(tokens)
stemmed = []
for token in tokens:
     stemmed_word = ps.stem(token)
     stemmed.append(stemmed_word)
print(stemmed)
df = pd.DataFrame(data={"tokens":tokens, "stemmed": stemmed})


# In[30]:


import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(x)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))


# In[31]:


import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(x)
for i in tokenized:
	wordsList = nltk.word_tokenize(i)
	wordsList = [w for w in wordsList if not w in stop_words]
	tagged = nltk.pos_tag(wordsList)
	print(tagged)
     


# In[ ]:




