import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, LabelEncoder, PolynomialFeatures
#from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis

train_df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktrain.csv")
test_df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktest.csv")

ndvi_cols = [col for col in train_df.columns if '_N' in col]

for df in [train_df, test_df]:
    df[ndvi_cols] = df[ndvi_cols].interpolate(axis=1).ffill(axis=1).bfill(axis=1)

def smooth_series(df, columns, window=3):
    return df[columns].T.rolling(window=window, min_periods=1, center=True).mean().T

train_df[ndvi_cols] = smooth_series(train_df, ndvi_cols)
test_df[ndvi_cols] = smooth_series(test_df, ndvi_cols)

def create_features(df):
    ndvi = df[ndvi_cols]
    features = pd.DataFrame()
    features['mean'] = ndvi.mean(axis=1)
    features['std'] = ndvi.std(axis=1)
    features['min'] = ndvi.min(axis=1)
    features['max'] = ndvi.max(axis=1)
    return features

X_train = create_features(train_df)
X_test = create_features(test_df)
y_train = train_df['class']

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.5, max_iter=800)
model.fit(X_train_scaled, y_train_encoded)

#y_pred_train = model.predict(X_train_scaled)
#print("Training Accuracy:", accuracy_score(y_train_encoded, y_pred_train))

y_test_preds = model.predict(X_test_scaled)
y_test_labels = le.inverse_transform(y_test_preds)

submission = pd.DataFrame({'ID': test_df['ID'], 'class': y_test_labels})
submission.to_csv("submission.csv", index=False)
