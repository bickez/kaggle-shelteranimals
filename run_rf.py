import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from process_data import process_data


df, df_features = process_data()

le = preprocessing.LabelEncoder()
target = le.fit_transform(df[df['train']]['OutcomeType'].values)
features = df_features[df['train']].values
clf = RandomForestClassifier(n_estimators=300)
forrest = clf.fit(features, target)

result = pd.DataFrame(forrest.predict_proba(df_features[df['train'] == False]), columns=le.classes_)
result.index += 1
result.to_csv('submissions/rf_out.csv', index_label='ID')
