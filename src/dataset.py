import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


df=pd.read_csv("Tweets.csv")
df = df.dropna()
df=df.reset_index(drop=True)

label_encoder=LabelEncoder()
df["label"]=label_encoder.fit_transform(df["sentiment"])

kf=KFold(n_splits=5)
df["fold"]=-1

for fold,(train_idx,val_idx) in enumerate(kf.split(X=df)):
    df.loc[val_idx,"fold"]=fold

train_df=df[df["fold"]!=0]
valid_df=df[df["fold"]==0]

