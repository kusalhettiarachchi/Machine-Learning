#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:12:03 2018

@author: rabbie
"""

#importing libraries
import pandas 
import numpy as np
import csv
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#opening dataset
#messages=[line.rstrip() for line in open('SMSSpamCollection')]
#print len(messages)

def SplitInToWords(message):
    message = unicode(message,'utf8')      
    return TextBlob(message).words    

def WordsInToBaseForm(message):
    words = TextBlob(unicode(message,'utf8').lower()).words
    return [word.lemma for word in words]

messages = pandas.read_csv('SMSSpamCollection', sep="\t",
                           quoting=csv.QUOTE_NONE, names=['label','message'])

#print messages.groupby('label').count()
#print messages.head() 
#print messages.message.head().apply(SplitInToWords)

trainingVector = CountVectorizer(analyzer=WordsInToBaseForm).fit(
        messages['message'])

message10 = trainingVector.transform([messages['message'][9]])
#print message10

#print messages['message'][9]
feature_names=trainingVector.get_feature_names()
#print feature_names
#print feature_names[5182]
#print trainingVector.get_feature_names()[3433]
messagesBagOfWords = trainingVector.transform(messages['message'])
messageTfidf = TfidfTransformer().fit(messagesBagOfWords).transform(messagesBagOfWords)

#training the model
spamFilter = MultinomialNB().fit(messageTfidf, messages['label'].values)

examples=['EnglandvMacedonia-dontmissthegoals/teamnews.Txt ENGLANDto99999',
          'Please call on this free line']

print spamFilter.predict(trainingVector.transform(examples))[1]




