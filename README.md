# age-friendly-busan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test = pd.read_csv('/content/sample_data/california_housing_test.csv')
train = pd.read_csv('/content/sample_data/california_housing_train.csv')
train.head()

test.head()

train.describe()

train.hist(figsize=(15,13) , grid=False , bins=50 )
plt.show()

correlation = train.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation , annot=True)
plt.show()
