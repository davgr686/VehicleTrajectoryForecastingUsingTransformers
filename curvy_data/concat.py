import pandas as pd
import glob
import os

df = pd.concat(map(pd.read_csv, glob.glob('*.csv')))
df.to_csv("curvy_data.csv")
print(df.describe())