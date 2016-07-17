import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

np.random.seed(33)
df = pd.read_csv('train_clean.csv')
df['has_name'] = np.where(df['has_name'], 1, 0)
df = df.fillna(df.mean())
print df.head()
print df.describe()
print pd.isnull(df)
le = preprocessing.LabelEncoder()
target = le.fit_transform(df['OutcomeType'].values)
features = pd.get_dummies(df[['RelativeAge', 'CleanColor', 'AnimalType', 'has_name']]).values
#imp = preprocessing.Imputer(missing_values='NaN')  # defaults to using 'mean'
#imp.fit(features)  # fit the imputer to the features
#feat_imp = imp.transform(features)  # transform the NaN values
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.7) # ~ 55% accuracy
clf = RandomForestClassifier(n_estimators=100)
y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print np.sum(np.diag(cm))*1. / np.sum(cm, axis=None)
print cm
print

clf = AdaBoostClassifier(n_estimators=100)
y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print np.sum(np.diag(cm))*1. / np.sum(cm, axis=None)
print cm
print

clf = GradientBoostingClassifier()
y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print np.sum(np.diag(cm))*1. / np.sum(cm, axis=None)
print cm
print