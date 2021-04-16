import pandas as pd


# Read in train, dev, and test datasets
df_train = pd.read_csv('./data/en-de/train.ende.df.short.tsv',sep="\t")
train = df_train[['original', 'translation', 'z_mean']]
train.head()