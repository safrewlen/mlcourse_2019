# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:14:03 2019

@author: Sergey Grechishkin
"""

import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

PATH_TO_DATA = 'data'
AUTHOR = 'Sergey_Grechishkin' 

def generate_words(train, test, id_to_site):
    sites = ['site%s' % i for i in range(1, 11)]

    train['words'] = train[sites].fillna(0).apply(lambda row: 
                                     ' '.join([id_to_site[i] for i in row]), axis=1).tolist()
    tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 5), stop_words=['unknown'])
    words_train = tfidf.fit_transform(train['words'])
    test['words'] = test[sites].fillna(0).astype(int).apply(lambda row: 
                                     ' '.join([id_to_site[i] for i in row]), axis=1).tolist()
    words_test = tfidf.transform(test['words'])
    
    test.drop(['words'], inplace=True, axis=1)
    train.drop(['words'], inplace=True, axis=1)
    
    return words_train, words_test

def generate_agg_func (data, func, func_name):
    time_cols = ['hour', 'hour_inter', '4hour', '4hour_inter',
       '2hour', '2hour_inter', '15min', '15min_inter', '30min', '30min_inter']
    for col in time_cols:
        data[func_name+'_'+col] = data.groupby(['date', col])['prediction'].transform(func)

def generate_time_intervals (train):
    train['hour']= train['time1'].apply(lambda ts: ts.hour)
    train['hour_inter']= train['time1'].apply(lambda ts: (ts.hour*60+ts.minute-30)//60)
    train['date'] = train['time1'].astype('datetime64[D]')
    train['4hour'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute)//240)
    train['4hour_inter'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute-120)//240)
    train['2hour'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute)//120)
    train['2hour_inter'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute - 60)//120)
    train['15min'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute)//15)
    train['15min_inter'] = train['time1'].apply(lambda ts: (ts.hour*60*60+ts.minute*60+ts.second - 450)//(15*60))
    train['30min'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute)//30)
    train['30min_inter'] = train['time1'].apply(lambda ts: (ts.hour*60+ts.minute-15)//30)
        
def cross_validation_oof (X_train, y_train, n_fold=5, seed=17):
   
    folds = StratifiedKFold(n_splits=n_fold)
  
    params = {
              'random_state':seed,
              'solver': 'liblinear',
   #           'n_jobs': -1,
             }

    scores = []

    oof = np.zeros(X_train.shape[0])
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):
    
        X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index] # train and validation data splits
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        model = LogisticRegression(**params)
        model.fit(X_train_fold, y_train_fold.values)
 
        preds = model.predict_proba(X_valid_fold)[:, 1]
        score = roc_auc_score(y_valid_fold, preds)

        oof[valid_index] = preds
        scores.append(score)
  
    print(scores)
    print(np.mean(scores), np.std(scores))
    return oof

cols = [
        'prediction',
        'mean_15min',
        'mean_15min_inter',
        'mean_30min',
        'mean_30min_inter',
        'mean_hour',
        'mean_hour_inter',
        'mean_4hour',
        'mean_4hour_inter',
        'mean_2hour',
        'mean_2hour_inter',
        'mean_day',
        

        'std_4hour',
        'std_4hour_inter',
        'std_30min',
        'std_30min_inter',
 
        'std_2hour',
        'std_2hour_inter',
        'std_hour',
        'std_hour_inter',  
        'std_day',
        
        '30mins_ave_mean',
    ]

times = ['time%s' % i for i in range(1, 11)]
sites = ['site%s' % i for i in range(1, 11)]
test = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'), parse_dates = times, index_col='session_id')
train = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'), parse_dates = times, index_col='session_id')
hosts = pd.read_pickle(os.path.join(PATH_TO_DATA, 'site_dic.pkl'))

id_to_site = {}

for key, item in hosts.items():
    if key.startswith('www'):
        ind = key.index('.')
        rel_key = key[ind+1:]
    else:
        rel_key = key
    id_to_site[item] = rel_key
    
id_to_site[0] = 'unknown'

generate_time_intervals(test)
generate_time_intervals(train)

train.sort_values(by='time1', inplace=True)

 
words_train, words_test = generate_words(train, test, id_to_site)
oof = cross_validation_oof(words_train, train.target, 20)
    
model = LogisticRegression(C=5, penalty='l2', random_state=17, solver='liblinear')
model.fit(words_train, train.target)
test_preds = model.predict_proba(words_test)[:, 1]
    
train['prediction'] = oof
test['prediction'] = test_preds

for data in [train, test]:
    generate_agg_func(data, np.mean, 'mean')
    generate_agg_func(data, np.std, 'std')
    generate_agg_func(data, len, 'len')
    data['30mins_ave_mean'] = (data['len_30min']*data['mean_30min']+data['len_30min_inter']*data['mean_30min_inter']
                       - data['len_15min']*data['mean_15min']) / \
                        (data['len_30min'] + data['len_30min_inter'] - data['len_15min'])
    data['mean_day'] = data.groupby(['date'])['prediction'].transform(np.mean)
    data['std_day'] = data.groupby(['date'])['prediction'].transform(np.std)
    
train_ = pd.DataFrame(index=train.index)
test_ = pd.DataFrame(index=test.index)
train_[cols] = train[cols].fillna(0).copy()
test_[cols] = test[cols].fillna(0).copy()

#cross_validation_2(train_, train.target)
    
model = RandomForestClassifier(random_state=17,
                          n_estimators=500,
                          max_depth=8,
                          min_samples_leaf=5,
  #                       max_features = 6,
                          n_jobs=-1,)
model.fit(train_, train.target)
y_test_rf = model.predict_proba(test_)[:, 1]

model = LogisticRegression(C = 5, random_state=17,  solver = 'liblinear')
model.fit(train_, train.target)
y_test_lr = model.predict_proba(test_)[:, 1]

y_test = 0.25*y_test_lr + 0.75*y_test_rf

submission = pd.DataFrame({"session_id": test.index, "target": y_test})
submission.to_csv(f'submission_alice_{AUTHOR}.csv', index=False)

