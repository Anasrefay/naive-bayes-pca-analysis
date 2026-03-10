import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath).replace('?', pd.NA)
    return df.fillna(df.mode().iloc[0])

def preprocess_catdata(df, target):
    X_enc = OrdinalEncoder().fit_transform(df.drop(columns=[target]))
    y_enc = LabelEncoder().fit_transform(df[target])
    return X_enc, y_enc

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)