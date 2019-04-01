# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:46:02 2019

@author: Kangwen
"""

import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


train = pd.read_csv('train.csv') #666615 datapoints
test = pd.read_csv('test.csv') #172402 datapoints

print(train.describe())

# =============================================================================
# ##### SPELLCHECK DATABASE #####
# words = open('words.txt').readlines()
# words = [word.strip() for word in words]
# =============================================================================

##### TRAIN-VALIDATION SPLIT #####

label = train['Category']
features = train.drop(['Category'], axis=1)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(features, label)

##### COUNT VECTOR AS FEATURE #####

# count vector is a matrix notation of the dataset in which every row represents a
# product title, every column is a term from the title, and every cell represents 
# the frequency count of a particular term

train_title = train['title']
test_title = test['title']
corpus = pd.concat([train_title, test_title])
vectorizer = CountVectorizer(analyzer = 'word', stop_words='english')
vectorizer.fit(corpus) #this creates the vectorizer
vocab = vectorizer.vocabulary_
#vocablist = list(vocab.keys()) 

#not_in_dict = set(vocablist) - set(words) 


#print(vectorizer.vocabulary_)
training_features = vectorizer.transform(train_x['title'])
validation_features = vectorizer.transform(valid_x['title'])
#xtraincountdf = pd.DataFrame(xtrain_count.toarray())

##### LOGISTIC REGRESSION #####
model = LogisticRegression()
model.fit(training_features, train_y)
pred_y = model.predict(validation_features)

acc = accuracy_score(valid_y, pred_y)

print("Accuracy using Logistic Regression: {:.2f}%".format(acc*100))

# =============================================================================
# ##### LINEAR SVC #####
# model = LinearSVC()
# model.fit(training_features, train_y)
# pred_y = model.predict(validation_features)
# 
# acc = accuracy_score(valid_y, pred_y)
# 
# print("Accuracy using Linear SVC: {:.2f}%".format(acc*100))
# =============================================================================

# =============================================================================
# ##### NAIVE BAYES #####
# model = MultinomialNB()
# model.fit(training_features, train_y)
# pred_y = model.predict(validation_features)
# 
# acc = accuracy_score(valid_y, pred_y)
# 
# print("Accuracy using Multinomial Naive Bayes: {:.2f}%".format(acc*100))
# =============================================================================

# =============================================================================
# ##### LOGISTIC REGRESSION #####
# 
# model_pipeline = Pipeline([('tfidf', TfidfTransformer()),
#                ('clf', LogisticRegression()),
#               ])
# 
# model_pipeline.fit(training_features, train_y)
# pred_y = model_pipeline.predict(validation_features)
# 
# acc = accuracy_score(valid_y, pred_y)
# 
# print("Accuracy using Logistic Regression: {:.2f}%".format(acc*100))
# 
# =============================================================================
