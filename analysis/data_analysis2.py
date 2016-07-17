import pandas as pd
import numpy as np
from collections import defaultdict
import re

df = pd.read_csv('train.csv')

df['OutcomeType'].value_counts()
df['AnimalType'].value_counts()
df[['OutcomeType', 'AnimalType']].describe()

def age_to_months(x):
	try: 
		spl = x.split(' ')
	except AttributeError:
		return np.nan
	if spl[1] == 'years' or spl[1] == 'year':
		return int(spl[0]) * 12
	elif spl[1] == 'weeks' or spl[1] == 'week':
		return int(spl[0]) / 4.
	elif spl[1] == 'days' or spl[1] == 'day':
		return int(spl[0]) / 30.
	else:
		return int(spl[0])	

df['AgeMonths'] = df['AgeuponOutcome'].apply(lambda x : age_to_months(x))
df[['AgeMonths', 'AnimalType']].groupby('AnimalType').mean()
mean_lifespan = df[['AgeMonths', 'AnimalType']].groupby('AnimalType').mean()
df['RelativeAge'] = np.where(df['AnimalType'] == 'Cat', df['AgeMonths'] / mean_lifespan.loc['Cat'].values, df['AgeMonths'] / mean_lifespan.loc['Dog'].values)
df['is_old'] = df['RelativeAge'] > 1.5

df['has_name'] = pd.isnull(df['Name'])



# order of colors with counts > 1000
df['SimpleColor'] = df['Color'].apply(lambda x : re.split('/| Tabby| ', x)[0].strip())

# add is_intact variable
df['is_intact'] = df['SexuponOutcome'].apply(lambda x: str(x).lower().find('intact') > -1)

# get datetime variables
dt = pd.to_datetime(df['DateTime'])

df['year'] = dt.apply(lambda x: x.year)
df['month'] = dt.apply(lambda x: x.month)
df['hour'] = dt.apply(lambda x: x.hour)
df['weekday'] = dt.apply(lambda x: x.weekday())

df['quick_breed'] = df['Breed'].apply(lambda x: re.split('/|Mix', x)[0].strip())
df['is_mix'] = df['Breed'].apply(lambda x: str(x).lower().find('mix') > -1)

print df.head()
df[['OutcomeType', 'AnimalType', 'has_name', 'RelativeAge', 'SimpleColor', 'is_intact', 'year', 'month', 'hour', 'weekday', 'quick_breed', 'is_old', 'is_mix']].to_csv('train_clean2.csv')




