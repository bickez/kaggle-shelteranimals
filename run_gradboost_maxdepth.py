import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

from process_data import process_data	

df, df_features = process_data()

le = preprocessing.LabelEncoder()
target = le.fit_transform(df[df['train']]['OutcomeType'].values)
features = df_features[df['train']].values
clf = GradientBoostingClassifier(n_estimators=500, max_depth=5)
grad = clf.fit(features, target)

result = pd.DataFrame(grad.predict_proba(df_features[df['train'] == False]), columns=le.classes_)
result.index += 1
result.to_csv('sumbissions/grad_boost_out_maxdepth_5.csv', index_label='ID')
