# encriptix-1-credit-card-fraud-detetion
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load the train and test datasets
train_data = pd.read_csv('fraudTrain[1].csv')
test_data = pd.read_csv('fraudTest[1].csv')

# Data preprocessing
# Handle missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Convert 'trans_date_trans_time' to datetime format
train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time'])
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])

# Extract features from 'trans_date_trans_time'
train_data['trans_year'] = train_data['trans_date_trans_time'].dt.year
train_data['trans_month'] = train_data['trans_date_trans_time'].dt.month
train_data['trans_day'] = train_data['trans_date_trans_time'].dt.day
train_data['trans_hour'] = train_data['trans_date_trans_time'].dt.hour
train_data['trans_minute'] = train_data['trans_date_trans_time'].dt.minute
train_data['trans_second'] = train_data['trans_date_trans_time'].dt.second

test_data['trans_year'] = test_data['trans_date_trans_time'].dt.year
test_data['trans_month'] = test_data['trans_date_trans_time'].dt.month
test_data['trans_day'] = test_data['trans_date_trans_time'].dt.day
test_data['trans_hour'] = test_data['trans_date_trans_time'].dt.hour
test_data['trans_minute'] = test_data['trans_date_trans_time'].dt.minute
test_data['trans_second'] = test_data['trans_date_trans_time'].dt.second

# Split the data
X_train = train_data.drop(['trans_date_trans_time', 'is_fraud'], axis=1)
y_train = train_data['is_fraud']
X_test = test_data.drop(['trans_date_trans_time', 'is_fraud'], axis=1)
y_test = test_data['is_fraud']

# Feature engineering
# Create new features
X_train['transaction_amount_log'] = X_train['amt'].apply(lambda x: np.log(x + 1))
X_test['transaction_amount_log'] = X_test['amt'].apply(lambda x: np.log(x + 1))

# Scale the features
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numeric_features)
    ])

# Fit and transform on training data
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform test data
X_test_scaled = preprocessor.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
print("Logistic Regression:")
print(f"Accuracy: {lr_accuracy:.2f}")
print(f"Precision: {lr_precision:.2f}")
print(f"Recall: {lr_recall:.2f}")
print(f"F1-score: {lr_f1:.2f}")

# Decision Trees
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
print("\nDecision Trees:")
print(f"Accuracy: {dt_accuracy:.2f}")
print(f"Precision: {dt_precision:.2f}")
print(f"Recall: {dt_recall:.2f}")
print(f"F1-score: {dt_f1:.2f}")

# Random Forests
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print("\nRandom Forests:")
print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1-score: {rf_f1:.2f}")
