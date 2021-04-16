import pandas as pd
from utils import fit, un_fit


# Read in train, dev, and test datasets
df_train = pd.read_csv('./data/en-de/train.ende.df.short.tsv',sep="\t")
train = df_train[['original', 'translation', 'z_mean']]
df_dev = pd.read_csv('./data/en-de/dev.ende.df.short.tsv',sep='\t')
dev = df_dev[['original', 'translation', 'z_mean']]
df_test = pd.read_csv('./data/en-de/test20.ende.df.short.tsv',sep='\t')
test = df_test[['original', 'translation', 'z_mean']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()

train = fit(train, 'labels')
dev = fit(dev, 'labels')