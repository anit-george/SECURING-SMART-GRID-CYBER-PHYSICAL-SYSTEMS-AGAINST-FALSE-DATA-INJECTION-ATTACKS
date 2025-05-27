
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['marker'] = LabelEncoder().fit_transform(df['marker'])  # Natural=1, Attack=0
    return df

def split_data(df):
    X = df.drop('marker', axis=1)
    y = df['marker']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
