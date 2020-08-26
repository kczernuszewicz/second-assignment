import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df_results = pd.read_csv('survey_results_public.csv',
                         usecols=['Respondent', 'Age1stCode', 'YearsCode', 'YearsCodePro'],
                         index_col='Respondent')

df_results.dropna(inplace=True)
df_results.replace(to_replace={'Younger than 5 years': '4',
                               'Older than 85': '86'},
                   inplace=True)
df_results.replace(to_replace={'Less than 1 year': '0',
                               'More than 50 years': '51'},
                   inplace=True)

df_results[['Age1stCode', 'YearsCode', 'YearsCodePro']] = df_results[
    ['Age1stCode', 'YearsCode', 'YearsCodePro']].astype(float)


print(df_results.corr())

Q1 = df_results.quantile(0.25)
Q3 = df_results.quantile(0.75)
IQR = Q3 - Q1

df_results = df_results[~((df_results < (Q1 - 1.5 * IQR)) | (df_results > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df_results.corr())

sns.boxplot(y='Age1stCode', data=df_results)
plt.show()

sns.boxplot(y='YearsCodePro', data=df_results)
plt.show()

sns.boxplot(y='YearsCodePro', data=df_results)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df_results[['YearsCodePro', 'Age1stCode']], df_results.YearsCode, test_size=0.2, random_state=777)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

reg_pred = reg.predict(X_test)

print(mean_squared_error(y_test, reg_pred))
