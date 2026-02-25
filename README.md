# Credit Card Fraud Detection 
A machine learning project designed to identify fraudulent credit card transactions.

#Project Overview

This project aims to detect fraudulent activities in credit card transactions using various machine learning algorithms.
**[Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)**

## About the Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. Key characteristics of the dataset include:

* **Highly Imbalanced Structure:** The number of legitimate transactions significantly outweighs the number of fraudulent ones. Fraud cases represent a very small percentage of the total data.
* **PCA Transformation:** To protect customer privacy, most features (columns V1 through V28) have been transformed using Principal Component Analysis (PCA).
* **Original Variables:** Only the `Time` and `Amount` features remain in their original form. Since the `Amount` variable contains extreme outliers, it was scaled using `RobustScaler` before modeling.


## Algorithms & Methodology
To handle the extreme class imbalance and achieve high predictive accuracy, the following machine learning algorithms were implemented:

* **Logistic Regression**
* **Random Forest Classifier**
* **XGBoost**

The models were evaluated using **Precision**, **Recall**, and **F1-Score** metrics—which are more appropriate for imbalanced data than standard **Accuracy**. Additionally, hyperparameter optimization was performed on the XGBoost model using `RandomizedSearchCV`.

## Project Visuals

### 1. Dataset Balance
![Sınıf Dağılımı](images/class_distribution.png)

### 2. Feature Correlation
![Korelasyon Analizi](images/correlation_heatmap.png)

### 3. Outlier Analysis
Veri setindeki uç değerlerin tespiti ve RobustScaler kullanımı öncesi analiz:
![Aykırı Değer Analizi](images/outlier_analysis.png)

### 4. Feature Importance
Eğitilen modelin en çok dikkat ettiği özellikler:
![Özellik Önemi](images/feature_importance.png)



