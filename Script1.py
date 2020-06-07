import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df_results = pd.read_csv('survey_results_public.csv',
                         usecols=['Respondent', 'Age1stCode', 'YearsCode', 'YearsCodePro', 'ConvertedComp', 'Age', 'Hobbyist', 'MgrIdiot'],
                         index_col='Respondent')
df_results.dropna(inplace=True)

df_results.replace(to_replace={'Younger than 5 years': '4',
                              'Older than 85': '86'},
                  inplace=True)

df_results.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)

df_results.replace(to_replace={'Yes': '1',
                              'No': '0'},
                  inplace=True)

labelencoder = LabelEncoder()
df_results['MgrIdiot'] = labelencoder.fit_transform(df_results['MgrIdiot'])
print(pd.unique(df_results['MgrIdiot'].values.ravel()))

df_results[['Age1stCode', 'YearsCode', 'YearsCodePro', 'Hobbyist']] = df_results[['Age1stCode', 'YearsCode', 'YearsCodePro', 'Hobbyist']].astype(float)

pd.set_option('display.max_columns', None)
print(df_results.corr())

var_y = df_results[['YearsCode']]
var_x1 = df_results[['YearsCodePro']]
var_x2 = df_results[['Age1stCode']]

df_adj = df_results[['YearsCode', 'YearsCodePro', 'Age1stCode']]

Q1 = df_adj.quantile(0.25)
Q3 = df_adj.quantile(0.75)
IQR = Q3 - Q1

df_adj_sd = df_adj[np.abs(df_adj - df_adj.mean()) <= 3*df_adj.std()]

df_adj_q = df_adj[~((df_adj < (Q1 - 1.5 * IQR)) | (df_adj > (Q3 + 1.5 * IQR))).any(axis=1)]




sns.boxplot(y='YearsCode', data=df_adj_sd)
plt.show()
sns.boxplot(y='YearsCode', data=df_adj_q)
plt.show()

sns.boxplot(y='YearsCodePro', data=df_adj_sd)
plt.show()
sns.boxplot(y='YearsCodePro', data=df_adj_q)
plt.show()

sns.boxplot(y='Age1stCode', data=df_adj_sd)
plt.show()
sns.boxplot(y='Age1stCode', data=df_adj_q)
plt.show()

print(df_adj_sd.corr())
print(df_adj_q.corr())
