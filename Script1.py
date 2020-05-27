import pandas as pd

df_results = pd.read_csv('survey_results_public.csv',
                         usecols=['Age1stCode', 'YearsCode', 'YearsCodePro', 'ConvertedComp', 'Age'])
df_results.dropna(inplace=True)

print(pd.unique(df_results[['Age1stCode']].values.ravel()))
print(pd.unique(df_results[['YearsCode']].values.ravel()))
print(pd.unique(df_results[['YearsCodePro']].values.ravel()))
print(pd.unique(df_results[['ConvertedComp']].values.ravel()))
print(pd.unique(df_results[['Age']].values.ravel()))

df_results.replace(to_replace={'Younger than 5 years': '4',
                              'Older than 85': '86'},
                  inplace=True)

df_results.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)


