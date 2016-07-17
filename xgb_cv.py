import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from process_data import process_data	

df, df_features = process_data()

le = preprocessing.LabelEncoder()
target = le.fit_transform(df[df['train']]['OutcomeType'].values)
features = df_features[df['train']].values
dtrain = xgb.DMatrix(features, label=target)

params = {'objective':'multi:softprob', 'eta':0.75, 'bst:max_depth':2, 'num_class':len(np.unique(target)), 'silent':1}
num_round = 10
result = xgb.cv(params, dtrain, num_round)
result.to_csv('xgb_cv_results.csv')


