import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from process_data import process_data	

df, df_features = process_data()

le = preprocessing.LabelEncoder()
target = le.fit_transform(df[df['train']]['OutcomeType'].values)
features = df_features[df['train']].values
clf = xgb.XGBClassifier(n_estimators=300, objective='multi:softmax')
grad = clf.fit(features, target)

result = pd.DataFrame(grad.predict_proba(df_features[df['train'] == False].values), columns=le.classes_)
result.index += 1
result.to_csv('submissions/xg_boost_out.csv', index_label='ID')
