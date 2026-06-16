# Customer Segmentation Dashboard

## Project Overview

This project uses Machine Learning and Customer Analytics techniques to segment customers based on:

* Age
* Annual Income
* Spending Score

The project applies K-Means Clustering to identify customer groups and provides actionable marketing recommendations through an interactive Streamlit dashboard.

---

## Features

### Data Preprocessing

* Missing Value Handling
* Outlier Detection
* Feature Scaling
* Data Cleaning

### Exploratory Data Analysis

* Correlation Heatmap
* Distribution Analysis
* Pair Plots
* Interactive Visualizations

### Customer Segmentation

* K-Means Clustering
* Elbow Method
* Silhouette Analysis
* PCA Visualization

### Dashboard

* Customer Segment Prediction
* KPI Cards
* Customer Search
* Segment Distribution Charts
* 3D Customer Visualization
* Downloadable Reports

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Streamlit
* Matplotlib
* Seaborn

---

## How to Run

```bash
pip install -r requirements.txt

python train.py

streamlit run app.py
```

---

## Results

The model segments customers into five meaningful customer groups:

* Premium Customers
* Budget Customers
* Loyal Customers
* Young High Spenders
* Occasional Buyers

These segments can be used for targeted marketing campaigns and business decision-making.
