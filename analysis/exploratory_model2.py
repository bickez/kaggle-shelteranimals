import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

np.random.seed(33)
df = pd.read_csv('train_clean2.csv')
df = df.fillna(df.mean())

le = preprocessing.LabelEncoder()
target = le.fit_transform(df['OutcomeType'].values)
features = pd.get_dummies(df[['RelativeAge', 'SimpleColor', 'AnimalType', 'has_name', 'is_intact', 'year', 'month', 'hour', 'weekday', 'quick_breed', 'is_old', 'is_mix']]).values
#imp = preprocessing.Imputer(missing_values='NaN')  # defaults to using 'mean'
#imp.fit(features)  # fit the imputer to the features
#feat_imp = imp.transform(features)  # transform the NaN values
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.7)
clf = RandomForestClassifier(n_estimators=100)
forrest = clf.fit(X_train, y_train)
y_pred = forrest.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print np.sum(np.diag(cm))*1. / np.sum(cm, axis=None) # ~ 68% accuracy
print cm
print
print le.classes_
print y_pred[:5]
print le.inverse_transform(y_pred[:5])
# test outputtting the results
result = pd.DataFrame(forrest.predict_proba(X_test), columns=le.classes_)
result.index += 1
result.to_csv('rf_out.csv', index_label='ID')
