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

params = {'objective':'multi:softprob', 'eta':1, 'bst:max_depth':2, 'num_class':len(np.unique(target)), 'silent':1, 'colsample_bytree':0.5}
num_round = 100
grad = xgb.train(params, dtrain, num_round)

dtest = xgb.DMatrix(df_features[df['train'] == False].values)
result = pd.DataFrame(grad.predict(dtest), columns=le.classes_)
result.index += 1
result.to_csv('submissions/xg_boost_out_params3.csv', index_label='ID')
