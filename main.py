import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

df = pd.read_csv("creditcard.csv")
print(df.head())
print(df.info())

print(df["Class"].value_counts())

print(df.isnull().sum())
print(df.describe())
print(df.shape)

print(df.duplicated().sum())
df = df.drop_duplicates()

sns.countplot(data=df, x="Class", hue="Class", palette=["blue", "red"])
plt.title("Normal-Fraud")
plt.show()

plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.1, linecolor='white')
plt.title("Corr", fontsize=16)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Amount"], color="royalblue")
plt.title("Distribution of Transaction Amounts and Outliers", fontsize=14)
plt.xlabel("Amount", fontsize=12)
plt.show()

X = df.drop("Class", axis = 1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic = LogisticRegression(class_weight="balanced", max_iter=2000)
logistic.fit(X_train_scaled,y_train)

y_pred = logistic.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred_rf))

imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=imbalance_ratio, 
    n_jobs=-1, 
    eval_metric="logloss"
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred_xgb))
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred_xgb))

param_dist = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=15,             
    scoring="f1",          
    cv=3,                  
    n_jobs=-1,
    verbose=1              
)

random_search.fit(X_train_scaled,y_train)
print(random_search.best_params_)

best_xgb = random_search.best_estimator_
y_pred_best = best_xgb.predict(X_test_scaled)
print(classification_report(y_test,y_pred_best))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_best))

importances = best_xgb.feature_importances_
features = X.columns

feature_df = pd.DataFrame({"Feature": features, "Importance": importances})
top_features = feature_df.sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=top_features)
plt.title("Top 10 Features", fontsize=14)
plt.show()