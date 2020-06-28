# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:14 2019

@author: craig
"""

import pandas as pd
import numpy as np
import nltk
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import lda
import re

#
# nltk.download('all') # ran this once to get it all


# Create band specific URLs.

domain = 'http://www.plyrics.com'
bands = ['menzingers', 'lawrencearms','ironchic', 'offwiththeirheads',
         'spraynard', 'nothington', 'dillingerfour', 'gaslightanthem',
         'hotwatermusic','bannerpilot','rvivr', 'redcityradio', 
         'againstme', 'americansteel', 'falcon', 'latterman',
         'jawbreaker', 'leatherface', 'smokeorfire', 'elway',
         'northlincoln', 'shorebirds', 'holymess', 'dearlandlord',
         'joycemanor', 'saintecatherines', 'flatliners', 'beachslang',
         'restorations', 'brokedowns', 'davehause', 
         'brendankelly', 'sundowner', 'deadtome']


def plyrics_extract(bands):
    '''
    Arguments:
        bands: list of bands
    Returns:
        list of Lyrics for every song for band in list 
    
    '''


    domain = 'http://www.plyrics.com'
    
    return


band_url = [domain+'/'+band[0]+'/'+band+'.html' for band in bands]

# Extract all song links from each band's specific page.

song_url = []
for url in band_url:
    r_url = requests.get(url)
    url_soup = BeautifulSoup(r_url.text)
    
    for link in url_soup.find_all('a'):
        if link.get('href') is not None and '/lyrics' in link.get('href'):
            song_url.append(domain+link.get('href'))


# This block of code will extract lyrics from a plyrics webpage
band ='menzingers'
song= 'iwasborn'#'midwesternstates'

#r = requests.get('http://www.plyrics.com/lyrics/'+band+'/'+song+'.html')
r = requests.get(song_url[88])
b = BeautifulSoup(r.text)

lyric_soup = b.find('h3').find_next_siblings(text=True)
lyrics = [lyric.strip() for lyric in lyric_soup]

lyric_string = ' '
lyric_string = lyric_string.join(lyrics)
lyric_string = lyric_string.replace('start of lyrics', '')
lyric_string = lyric_string.replace('end of lyrics', '').strip()
lyric_string = lyric_string.replace('â', '\'')

# and at some point concat this in ot a big string (all the songs)
# or add to a list

#print(lyric_string)

lyric_list = []

for s_url in song_url:
    r = requests.get(s_url)
    b = BeautifulSoup(r.text)#.decode('utf-8', 'ignore')

    lyric_soup = b.find('h3').find_next_siblings(text=True)
    lyrics = [lyric.strip() for lyric in lyric_soup]

    lyric_string = ' '
    lyric_string = lyric_string.join(lyrics)
    lyric_string = lyric_string.replace('start of lyrics', '')
    lyric_string = lyric_string.replace('end of lyrics', '').strip()
    lyric_string = lyric_string.replace('â', '\'')
    lyric_string = lyric_string.replace('\x80\x99', '')
    lyric_list.append(lyric_string)

# left off here  'http://www.plyrics.com/lyrics/lawrencearms/turnstyles.html'
    # hopefully not banned forever
    

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import lda
import re

text = ' '.join(lyric_list)

tokens = [word for word in nltk.word_tokenize(text)]
tokens[:100]

clean_tokens = [token for token in tokens if re.search('^[a-zA-Z]+', token)]
clean_tokens[:100]

stopwords = nltk.corpus.stopwords.words('english')



sentences = [sent for sent in nltk.sent_tokenize(text)]
sentences[:10]

vect = CountVectorizer(stop_words=stopwords, ngram_range=[1,3]) 
sentences_train = vect.fit_transform(sentences)

# Instantiate an LDA model
model = lda.LDA(n_topics=12, n_iter=500, random_state=1)
model.fit(sentences_train) # Fit the model 
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vect.get_feature_names())[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ', '.join(topic_words)))

def maximize_loglike(ntopics=20, topspace=5):
    loglike = []
    for i in range(5,ntopics,topspace):
        temp_mod = lda.LDA(n_topics=i, n_iter=500, random_state=1)
        temp_mod.fit(sentences_train) # Fit the model 
        loglike.append((i, temp_mod.loglikelihood()))
        
    return loglike
# 5 is best topics
    
# len(lyric_list) is 151 songs
    
vect = CountVectorizer(stop_words=stopwords, ngram_range=[1,3]) #token_pattern 
X = vect.fit_transform(lyric_list)
model_2 = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model_2.fit(X) # Fit the model 
n_top_words = 10
topic_word = model_2.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vect.get_feature_names())[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ', '.join(topic_words)))