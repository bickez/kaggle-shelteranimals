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

df['AgeMonths'] = df['AgeuponOutcome'].apply(lambda x: age_to_months(x))
df[['AgeMonths', 'AnimalType']].groupby('AnimalType').mean()
mean_lifespan = df[['AgeMonths', 'AnimalType']].groupby('AnimalType').mean()
df['RelativeAge'] = np.where(df['AnimalType'] == 'Cat', df['AgeMonths'] / mean_lifespan.loc['Cat'].values, df['AgeMonths'] / mean_lifespan.loc['Dog'].values)

df['has_name'] = pd.isnull(df['Name'])

# figure out color frequencies of various basic colors
temp = df['Color'].apply(lambda x : re.split('/| Tabby| ', x))
temp_flat = np.hstack(temp.values)
un = np.unique(temp_flat)[1:]  # get rid of ''
# count different colors
d = defaultdict(int)
for w in temp_flat:
	d[w] += 1
print d
print un # unique colors

# order of colors with counts > 1000
l = ['White', 'Black', 'Brown', 'Tan', 'Blue', 'Brindle']
res_color = []
for c in df['Color']:
	ctemp = re.split('/| Tabby| ', c)
	loc = []
	for ct in ctemp:
		if ct in l:
			loc.append(l.index(ct))
	if len(loc) > 0:
		res_color.append(l[min(loc)])
	else:
		res_color.append('Other')

df['CleanColor'] = res_color
print df.head()
df[['OutcomeType', 'AnimalType', 'has_name', 'RelativeAge', 'CleanColor']].to_csv('train_clean.csv')



