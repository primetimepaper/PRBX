import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import zeros as z

csv_concat = "interpolated_concat.csv"
csv = "interpolated.csv"

df = pd.read_csv(csv, header=None)
df = df[6].to_list()
pred = z(len(df))
print("bare minimum for hmb1")
print(sqrt(mean_squared_error(df, pred)))

df = pd.read_csv(csv_concat, header=None)
df = df[6].to_list()
pred = z(len(df))
print("bare minimum for hmb_concat")
print(sqrt(mean_squared_error(df, pred)))
print("OK")