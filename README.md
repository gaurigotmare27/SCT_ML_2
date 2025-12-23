# SCT_ML_2
# Customer Segmentation using K-Means Clustering 🛍️

## 📌 Project Overview
This project implements a **K-Means Clustering** algorithm to group customers of a retail store based on their purchase history. The goal is to identify distinct customer segments to enable targeted marketing strategies.

## 📂 Dataset
* **Source:** [Mall Customer Segmentation Data (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
* **File:** `Mall_Customers.csv`
* **Features Used:**
  * Annual Income (k$)
  * Spending Score (1-100)

## 🛠️ Technologies Used
* **Python**
* **Scikit-Learn** (Clustering Model)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)

## 📊 Results
Using the **Elbow Method**, we determined the optimal number of clusters to be **5**. The algorithm identified these distinct customer profiles:
1. **Low Income, Low Spending**
2. **Low Income, High Spending**
3. **High Income, Low Spending**
4. **High Income, High Spending** (Target Audience)
5. **Average Income, Average Spending**

## 💻 Code Snippet
Here is the core logic used to create the clusters:

```python
from sklearn.cluster import KMeans

# 1. Train the model with 5 clusters (determined by Elbow Method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 2. Predict a new customer's segment
# Example: Income = $90k, Spending Score = 15
new_customer = [[90, 15]]
predicted_cluster = kmeans.predict(new_customer)
print(f"The new customer belongs to Cluster: {predicted_cluster[0]}")
