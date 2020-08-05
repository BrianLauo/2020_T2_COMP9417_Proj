import os
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from ast import literal_eval
import warnings
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings('ignore')

data_path = './dataset/'

def read_df(path, file_name, nrows=None):
    df = pd.read_csv(path + file_name, dtype={'fullVisitorId': 'str', 'visitId': 'str'}, chunksize=nrows)
    return df

train_df = read_df(data_path, 'train.csv')
test_df = read_df(data_path, 'test.csv')

# Drop
train_df = train_df.drop(['visitId', 'visitStartTime', 'campaignCode'], axis=1)
test_df = test_df.drop(['visitId', 'visitStartTime'], axis=1)

# Fill NA
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# Encode non-numerics
non_numerics = train_df.select_dtypes(exclude=np.number).columns.tolist()
for col in non_numerics:
    if col in ['fullVisitorId']: continue
    print(col)
    lb = preprocessing.LabelEncoder()
    lb.fit(list(train_df[col].values) + list(test_df[col].values))
    train_df[col] = lb.transform(list(train_df[col].values))
    test_df[col] = lb.transform(list(test_df[col].values))
    
# Split DF
train_x = train_df.drop(['fullVisitorId', 'totalTransactionRevenue','index','campaign'], axis = 1)
train_y = np.log1p(train_df['totalTransactionRevenue'].values)
trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
test_x = test_df.drop(['fullVisitorId', 'totalTransactionRevenue','index','campaign'], axis = 1)
test_y = np.log1p(test_df['totalTransactionRevenue'].values)

# Transform to lgb dataset
train_data = lgb.Dataset(trn_x, label = trn_y)
test_data = lgb.Dataset(val_x, label = val_y, reference = train_data)

# Model parameters
parameters = {
    'objective': 'regression',
    'metric': 'rmse',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 35,
    'feature_fraction': 0.3,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.1,
}

# Train
model = lgb.train(parameters, train_data, valid_sets=test_data, num_boost_round=300, early_stopping_rounds=100)

# predict
preds = model.predict(test_x, num_iteration=model.best_iteration)

# Generate submission file
preds[preds < 0] = 0
submission = test_df[['fullVisitorId']]
submission['PredictedLogRevenue'] = np.expm1(preds)
submission = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
submission['PredictedLogRevenue'] = np.log1p(submission['PredictedLogRevenue'])
submission.to_csv('submission.csv', index=False)