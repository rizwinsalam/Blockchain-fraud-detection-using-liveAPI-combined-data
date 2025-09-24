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

<img width="562" height="467" alt="Fraud_normal" src="https://github.com/user-attachments/assets/b7353f5b-170c-4695-a578-2240bc101e93" />

PCA
<img width="790" height="590" alt="pca_kmeans" src="https://github.com/user-attachments/assets/72b186ba-f77d-4719-91be-e49698863dcd" />

<img width="543" height="435" alt="PCA_Kmeans (2)" src="https://github.com/user-attachments/assets/bb853205-5d4a-4c88-8d60-af6387e0c4a7" />


4️⃣ Transaction Value Distribution
<img width="990" height="590" alt="distbtn_of_trnsctn_value" src="https://github.com/user-attachments/assets/e8169f4b-12c4-45a9-b477-51d4922c247e" />


5️⃣ Top sender
<img width="989" height="590" alt="topsender" src="https://github.com/user-attachments/assets/940f2902-21d8-47e2-bf85-b0b6d31447fe" />

Top Receiver
<img width="989" height="590" alt="topreceiver" src="https://github.com/user-attachments/assets/266cef63-d9ae-4a59-a5c7-50eb5285f119" />


6️⃣ Monthly Transaction Trends
<img width="988" height="547" alt="monthlytrnsctncounts" src="https://github.com/user-attachments/assets/e3bbeec5-0da0-4fb7-97df-d3cf72627d5a" />

7️⃣ Scatter Plot: Gas vs Value
<img width="989" height="590" alt="tarnsctn_value_vs_gasfee" src="https://github.com/user-attachments/assets/fccdc1aa-4859-45c7-9e98-a2e022f9375e" />
8️⃣ Correlation Heatmap
<img width="483" height="390" alt="heatmap" src="https://github.com/user-attachments/assets/b1ebeb4e-1fab-4621-a0c8-dfd82e00c16e" />

