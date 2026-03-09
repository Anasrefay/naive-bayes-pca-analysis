import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    
    data = data.replace('?', pd.NA)
    data = data.fillna(data.mode().iloc[0]) 
    
    return data

def preprocess_catdata(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    feature_encoder = OrdinalEncoder()
    X_encoded = feature_encoder.fit_transform(X)
    
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X_encoded, y_encoded, feature_encoder, target_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)