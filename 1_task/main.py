import pandas as pd
import os
from matplotlib import pyplot as plt

def main():
  my_path = os.path.dirname(os.path.abspath(__file__))
  df = pd.read_csv(my_path + '/PASTA_PURCHASE.csv')
  # Exercise 1
  print('Mean = ' + str(df['PASTA'].mean()) + ' Std = ' + str(df['PASTA'].std()))
  # Exercise 2
  print(df.groupby('AREA')['INCOME'].mean())
  # Exercise 3
  print("Max values \n ", str(df.groupby('AREA')['INCOME'].max()))
  print("Min values \n ",  str(df.groupby('AREA')['INCOME'].min()))
  print(df.groupby('HHID')['PASTA'].sum().max())
  #Exercise 4
  filter1 = df['AREA'] == 2
  filter2 = df['INCOME'] > 20000.0
  df = pd.read_csv(my_path + '/PASTA_PURCHASE.csv')
  df.where(filter1 & filter2, inplace = True)
  df.dropna(inplace = True)
  df = df.groupby('HHID')['PASTA'].sum()>30
  print(df.value_counts())
  # Exercise 5
  df_corr = pd.read_csv(my_path + '/PASTA_PURCHASE.csv')
  print(df_corr[['PASTA','EXPOS']].corr(method = 'pearson'))
  # Exercise 6
  df_corr.groupby('TIME')['PASTA'].sum().reset_index().plot(kind = 'scatter', x = 'TIME', y = 'PASTA')
  plt.xlabel('TIME')
  plt.ylabel('PASTA')
  plt.show()


if __name__ == '__main__':
    main()