# =======================
# SILKY SKY AIRWAYS - TASK 1
# Logistic Regression Model for Passenger Satisfaction
# =======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Optional: For reproducibility ---
import random
random.seed(42)
np.random.seed(42)

# --- File Upload Dialogue ---
try:
    from google.colab import files
    print("Please upload SILKYSKY_DATA_CW2.csv...")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
except ImportError:
    # For local environment (non-Colab)
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(title="Select SILKYSKY_DATA_CW2.csv file")
    print(f"File selected: {filename}")

# --- Load Data ---
df = pd.read_csv(filename, encoding='latin1')

# Display basic info about the dataset
print("Dataset Shape:", df.shape)
print("Data Types and Non-Null Counts:\n", df.info())
print("Total records in original dataset:", len(df))

# Check for missing values
missing = df.isnull().sum()
print("\nMissing Values per Column:\n", missing[missing > 0])

# Summary statistics for numerical columns
print("\nSummary Statistics:\n", df.describe())

# --- Visualization 1: Satisfaction Distribution ---
sns.countplot(x='Satisfied', data=df)
plt.title('Passenger Satisfaction Distribution')
plt.show()

# --- Visualization 2: Age Distribution ---
plt.figure(figsize=(12,6))
sns.countplot(x='Age', data=df, order=df['Age'].value_counts().index)
plt.title('Passenger Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# --- Visualization 3: Correlation Heatmap ---
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# --- Data Preprocessing ---

# 1. Convert 'Satisfied' to binary 0/1
df['Satisfied'] = df['Satisfied'].map({'Y': 1, 'N': 0})

# 2. Label encode categorical columns
le = LabelEncoder()
categorical_cols = ['Gender', 'Type of Travel', 'Class', 'Age Band', 'Destination', 'Continent']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Define features and target
X = df.drop(['Ref', 'id', 'Satisfied'], axis=1)
y = df['Satisfied']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Handle missing values
print("Before dropna:", X_train.shape, y_train.shape)
train_data = X_train.copy()
train_data['target'] = y_train
train_data = train_data.dropna()
print("After dropna:", train_data.shape)

X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

# Add class distribution after cleaning
print("\nTrain Class Distribution (after dropna):")
print(y_train.value_counts())

# Fill missing in test data
median_values = X_train.median()
X_test = X_test.fillna(median_values)
y_test = y_test.loc[X_test.index]
mask = y_test.notna()
X_test = X_test.loc[mask]
y_test = y_test.loc[mask]

# 6. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train model with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 8. Evaluate
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- NEW: Confusion Matrix Plot ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Satisfied', 'Satisfied'],
            yticklabels=['Not Satisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


feature_names = X_train.columns
coefficients = model.coef_[0]
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
print(coef_df)


from sklearn.metrics import roc_curve, roc_auc_score

y_proba = model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


X_clean = X.dropna()
y_clean = y.loc[X_clean.index]
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, scaler.transform(X_clean), y_clean, cv=5, scoring='accuracy')
print("5-fold CV Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
# new thign

from sklearn.ensemble import RandomForestClassifier

# --- Train Random Forest model ---
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

# --- Predictions and evaluation ---
y_pred_rf = rf_model.predict(X_test_scaled)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:,1]

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix for RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Not Satisfied', 'Satisfied'],
            yticklabels=['Not Satisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.show()

# ROC Curve and AUC for RF
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# Feature importances from Random Forest
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
print("\nRandom Forest Feature Importances:")
print(feat_imp_df)

# --- Cross-validation for Random Forest ---
cv_scores_rf = cross_val_score(rf_model, scaler.transform(X_clean), y_clean, cv=5, scoring='accuracy')
print("Random Forest 5-fold CV Accuracy Scores:", cv_scores_rf)
print("Random Forest Mean CV Accuracy:", cv_scores_rf.mean())

# --- Summary comparison table ---
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Test Accuracy': [accuracy_score(y_test, y_pred), accuracy_score(y_test, y_pred_rf)],
    'Mean CV Accuracy': [cv_scores.mean(), cv_scores_rf.mean()],
    'Test ROC AUC': [auc, auc_rf]
})

print("\nModel Performance Comparison:")
print(comparison_df)

