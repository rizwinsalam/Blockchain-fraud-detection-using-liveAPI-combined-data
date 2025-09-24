# Blockchain-fraud-detection-project
Ethereum transactions anomaly detection with live API integration and data visualization using Python.
# 🚨 Blockchain Fraud Detection Project

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview
This project demonstrates *fraud detection on Ethereum blockchain transactions* using Python and data visualization.  
We simulate fraudulent transactions (10–20% of total) to analyze patterns, anomalies, and trends.

*Highlights:*
- Transaction analysis (top senders/receivers)  
- Transaction value distribution  
- Time-based trends and anomalies  
- Visualizations: bar charts, histograms, pie charts, scatterplots, heatmaps

*Technologies Used:*
Python (Pandas, NumPy, Matplotlib, Seaborn), Jupyter Notebook, optional MySQL integration.

---

## 📊 Dataset
*Columns:*

| Column        | Description                        |
|---------------|------------------------------------|
| from        | Sender Ethereum address             |
| to          | Receiver Ethereum address           |
| value_ETH   | Transaction value in ETH            |
| timestamp   | Transaction date & time             |
| is_fraud    | Fraud label (0 = Normal, 1 = Fraud)|

> is_fraud labels are *simulated* for demonstration purposes.
---
📈 Exploratory Data Analysis (EDA)
1️⃣ Fraud vs Normal Transactions



2️⃣ Top 10 channels


3️⃣ Top 10 Receivers


4️⃣ Transaction Value Distribution


5️⃣ Fraud Share by Value


6️⃣ Monthly Transaction Trends



7️⃣ Scatter Plot: Gas vs Value

8️⃣ Correlation Heatmap
