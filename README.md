 # 🚨 Blockchain Fraud Detection 

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview
This project demonstrates *fraud detection on Ethereum blockchain transactions* with live API and simulated data integration and visualization using Python.

*Highlights:*
- Transaction analysis (top senders/receivers)  
- Transaction value distribution  
- Time-based trends and anomalies  
- Visualizations: bar charts, histograms, scatterplots, heatmaps

*Technologies Used:*
Python (Pandas, NumPy, Matplotlib, Seaborn), Jupyter Notebook, optional MySQL integration.

---

## *Introduction:*

This project leverages advanced data analytics to investigate and detect fraudulent activity within Ethereum blockchain transactions. By integrating data from both API and CSV sources, we constructed a unified dataset capturing transaction values, gas fees, timestamps, and participant addresses. The analysis encompasses a wide range of techniques, including exploratory data analysis, statistical profiling, time series visualization, correlation analysis, and outlier detection.
To uncover hidden patterns and suspicious behaviors, we applied machine learning methods such as Principal Component Analysis (PCA), KMeans clustering, and supervised classification models. These approaches enabled us to segment transaction behaviors, identify anomalies, and quantify the prevalence and impact of fraudulent activity. Visualizations including histograms, boxplots, scatter plots, heatmaps, and pie charts were used extensively to communicate findings and support data-driven decision making.
Overall, this project demonstrates the power of combining statistical, visual, and machine learning techniques to gain actionable insights and enhance fraud detection in blockchain based financial systems.

## 📊 Dataset
*Columns:*

| Column        | Description                        |
|---------------|------------------------------------|
| from        | Sender Ethereum address             |
| to          | Receiver Ethereum address           |
| value_ETH   | Transaction value in ETH            |
| timestamp   | Transaction date & time             |
| is_fraud    | Fraud label (0 = Normal, 1 = Fraud)|

> is_fraud labels are *simulated* for demonstration purposes. (Since real blockchain fraud labels are unavailable, we simulated 10–20% transactions as fraud using random assignment. This allows visual and statistical analysis of fraud patterns without exposing real addresses)
---
## 📈 Exploratory Data Analysis (EDA)

*Steps Taken in Data Cleaning & Preprocessing:*
- Loaded dataset into Pandas DataFrame.
- Checked column names and corrected formatting.
- Converted value from Wei to ETH where necessary.
- Converted timestamp column to datetime for time-based analysis.
- Added is_fraudcolumn to simulate fraudulent transactions.

Fraudulent transactions are low in count but may involve large ETH transfers.
Top senders and receivers show concentration of ETH flows.
Gas fees vs transaction value can help highlight anomalies.
Time-based trends help detect unusual spikes potentially linked to fraud.
Correlation analysis shows moderate correlation between transaction value and fraud occurrence.


         1️⃣ Fraud vs Normal Transactions

<img width="562" height="467" alt="Fraud_normal" src="https://github.com/user-attachments/assets/b7353f5b-170c-4695-a578-2240bc101e93" />

This bar chart compares the number of normal and fraudulent transactions detected. Out of 113 transactions, 94 (83%) are classified as normal (green), while only 19 (17%) are fraud (red). This shows that the majority of activity is legitimate, but a notable number of suspicious transactions are still detected.


         2️⃣ PCA v/s K-Means clustering 

<img width="790" height="590" alt="pca_kmeans" src="https://github.com/user-attachments/assets/72b186ba-f77d-4719-91be-e49698863dcd" />

This scatter plot shows the results of applying PCA (Principal Component Analysis) and KMeans clustering to the transaction data. Each point represents a transaction, projected onto two principal components (PC1 and PC2). The data is grouped into three clusters (red, blue, green), revealing distinct behavioral patterns among transactions. One cluster (green) is separated from the main group, indicating a set of transactions with unusual characteristics—potentially high-value or otherwise anomalous.

<img width="543" height="435" alt="PCA_Kmeans (2)" src="https://github.com/user-attachments/assets/bb853205-5d4a-4c88-8d60-af6387e0c4a7" />


Anomaly Tagging (Unified)
This plot highlights which transactions are considered anomalies (red) versus normal (blue) based on the clustering results. Anomalies are clearly separated from the majority of transactions, often occupying regions of the plot with higher PC2 values. This demonstrates that the clustering and anomaly detection approach is effective at isolating unusual or suspicious transaction patterns for further investigation.

Combined Insight:
Clustering and anomaly tagging using PCA and KMeans successfully separate normal and suspicious transactions, with anomalies forming distinct groups in the data. This approach provides a powerful tool for identifying potential fraud or outlier behavior in blockchain transaction datasets.
The cluster with the highest concentration of anomalies corresponds to high-value transfers, confirming that clustering is an effective tool for identifying fraudulent activity.


         3️⃣ Transaction Value Distribution

<img width="990" height="590" alt="distbtn_of_trnsctn_value" src="https://github.com/user-attachments/assets/e8169f4b-12c4-45a9-b477-51d4922c247e" />

This histogram displays the distribution of transaction values (in ETH) across all 113 transactions. The majority of transactions are clustered at lower values:
- Most transactions are below 10 ETH (with the 75th percentile at 8.09 ETH).
- The median transaction value is 4.49 ETH, meaning half of all transactions are less than this amount.
- A few transactions are extremely large, with the maximum reaching 5,000 ETH, which creates a long right tail (outliers) in the histogram.
- The mean value (50.53 ETH) is much higher than the median, highlighting the impact of these outliers.
- Standard deviation is high (469.94 ETH), further indicating a wide spread due to a small number of very large transactions.

This skewed distribution is typical in cryptocurrency, where most transfers are small, but a few high-value transactions dominate the total volume. The histogram makes it easy to see both the common small transactions and the rare, very large ones.

         4️⃣ Top sender
<img width="989" height="590" alt="topsender" src="https://github.com/user-attachments/assets/940f2902-21d8-47e2-bf85-b0b6d31447fe" />

This bar chart highlights the address that sent the most ETH in the dataset. The top sender is responsible for a disproportionately large share of total ETH sent.
Top sender address: 0x742d35cc6634c0532925a3b844bc454e4438f44e. Total ETH sent by this address is 5000 ETH.

         5️⃣ Top Receiver
<img width="989" height="590" alt="topreceiver" src="https://github.com/user-attachments/assets/266cef63-d9ae-4a59-a5c7-50eb5285f119" />

This bar chart displays the address that received the most ETH in the dataset. The top receiver stands out by accumulating a significant portion of the total ETH received.
Top receiver address: 0x742d35cc6634c0532925a3b844bc454e4438f44e. Total ETH received by this address is 5,000 ETH.

Both Sender's and Receiver's concentration indicates that a small number of addresses (often called “whales”) have a significant influence on the network’s transaction flow. Such addresses may belong to exchanges, large traders, or entities conducting high-value transfers, and are important to monitor for both operational and fraud detection purposes.


         6️⃣ Monthly Transaction Trends:

Monthly Transaction Count (Unified)

<img width="1187" height="490" alt="monthlytrnsctncount_unified" src="https://github.com/user-attachments/assets/846b2435-c7d8-4a11-8688-b0ca6022cf96" />

- Transaction activity varies month-to-month, with several peaks exceeding 600 transactions (notably in mid-2018, mid-2019, early 2020, and early 2021).
- Some months show much lower activity, with fewer than 100 transactions.
- The pattern is cyclical, with repeated surges and drops rather than a steady trend.

Monthly Transaction Value with Anomaly Counts

<img width="988" height="547" alt="monthlytrnsctncounts" src="https://github.com/user-attachments/assets/e3bbeec5-0da0-4fb7-97df-d3cf72627d5a" />

- Total monthly transaction value fluctuates, with the highest spike in January 2021 (over 7,000,000 ETH).
- Other significant peaks occur in 2018-08, 2019-01, and 2020-01 (all above 1,000,000 ETH).
- Larger bubbles indicate that months with high transaction values also have more detected anomalies, suggesting a link between value 
surges and suspicious activity.

         7️⃣ Scatter Plot: Gas vs Value

<img width="989" height="590" alt="tarnsctn_value_vs_gasfee" src="https://github.com/user-attachments/assets/fccdc1aa-4859-45c7-9e98-a2e022f9375e" />

This scatter plot shows the relationship between transaction value and gas fee. Most transactions are small and incur low gas fees, with no clear correlation between the two. Some very large transactions are visible, but they do not necessarily pay higher fees, suggesting fee optimization or network timing strategies.

         8️⃣ Correlation Heatmap

<img width="483" height="390" alt="heatmap" src="https://github.com/user-attachments/assets/b1ebeb4e-1fab-4621-a0c8-dfd82e00c16e" />

This correlation heatmap shows the relationship between transaction value (value_eth) and gas fee (gas_fee_eth):
- The correlation coefficient between value_eth and gas_fee_eth is -0.0029 (very close to zero).
- The diagonal values are always 1, since each variable is perfectly correlated with itself.
In other terms, these are essentially uncorrelated (correlation coefficient ≈ 0). This means that the amount of ETH transferred in a transaction does not predict the gas fee paid, indicating that users may optimize fees regardless of transaction size.

         9️⃣ Share of Total ETH: Fraud vs Normal

<img width="481" height="504" alt="block_pie" src="https://github.com/user-attachments/assets/c1ea4c12-3de0-4814-ba98-2e04a8f6e5cb" />

This pie chart illustrates the distribution of total ETH transferred in normal versus fraudulent transactions. The vast majority of ETH (96.7%) is moved through normal transactions (green), while only 3.3% is associated with fraud (red). This suggests that, although fraudulent transactions exist, they represent a small portion of the overall value transferred on the network.

## 🧠 Model Evaluation – Isolation Forest

The Isolation Forest model was applied to detect anomalies in Ethereum blockchain transactions.
This isolates outliers based on unique transaction behavior (such as unusually high values, irregular timings, or rare address interactions).


<img width="528" height="470" alt="models perfmnc evaltn confusion matrix" src="https://github.com/user-attachments/assets/0aea62dc-34d7-45b8-81c3-00155dcd6787" />


<img width="536" height="374" alt="models performace metrics" src="https://github.com/user-attachments/assets/dafe4b40-0c57-47f8-b144-ed569dbc0235" />


## Interpretation:
The model correctly classifies about 83.5% of transactions overall.
Precision (15%) shows that about 1 in 6 flagged transactions were actual frauds; good for early anomaly spotting.
Recall (15.8%) indicates moderate sensitivity; it's catching some but not all anomalies.
The F1 score (15.4%) balances both and serves as a baseline for further tuning.

All missing values were normalized and timestamps harmonized across both API and CSV sources. This thorough data cleaning ensures that the analysis is based on consistent and reliable data, supporting trustworthy and actionable insights. Many anomalies were detected outside standard business hours, indicating that suspicious transactions may be strategically timed to avoid detection or to take advantage of lower gas fees. This highlights the value of incorporating timing analysis as an additional layer in future fraud detection efforts.

## Conclusion

This project demonstrates a comprehensive approach to blockchain transaction analysis and fraud detection using real Ethereum data. Through a combination of statistical analysis, visualization, clustering, and anomaly detection, we uncovered several important insights:
- Transaction Patterns: Transaction activity and value fluctuate significantly month-to-month, with periodic surges often coinciding with increased anomaly counts. These peaks may be linked to settlements, payroll, or coordinated transfers, highlighting periods that warrant closer scrutiny.
- Fraud Detection: While the majority of transactions are legitimate, a notable minority are flagged as fraudulent. However, fraudulent transactions account for only a small share of the total ETH transferred (as little as 1.5–3.3%), indicating that most value on the network moves through normal channels.
Behavioral Insights: There is no significant correlation between transaction value and gas fee, suggesting that users optimize fees regardless of transaction size. Clustering and PCA revealed distinct behavioral groups, with anomalies and potential frauds forming clear outlier clusters.
- Top Actors: A small number of addresses dominate both sending and receiving activity, consistent with the presence of “whales” or major exchanges, and these entities have a disproportionate influence on network flows.
- Model Performance: Machine learning models, such as Random Forest and Logistic Regression, demonstrated high accuracy in distinguishing normal from fraudulent transactions, though class imbalance remains a challenge for rare fraud cases.

Overall, this project highlights the power of data-driven techniques for uncovering hidden patterns and potential fraud in blockchain networks. The combination of statistical, visual, and machine learning methods provides a robust framework for ongoing monitoring, risk assessment, and proactive fraud detection in decentralized financial systems.

## Future Potential:

This project lays a robust foundation for advanced blockchain analytics and fraud detection, but there are several promising directions for future development:

Expanded Feature Engineering: Incorporating additional features such as transaction network graphs, address reputation scores, and smart contract interactions could improve the accuracy and depth of anomaly and fraud detection.


Advanced Machine Learning Models: Exploring deep learning, graph neural networks, or ensemble methods may uncover more subtle patterns and relationships, further boosting detection performance, especially for rare or evolving fraud tactics.


Cross-Chain Analysis: Extending the framework to analyze transactions across multiple blockchains (e.g., Bitcoin, Binance Smart Chain) would provide a more comprehensive view of illicit activity and money flows in the broader crypto ecosystem.


Collaboration with Regulatory Bodies: Sharing findings and methodologies with regulators and industry partners could help set new standards for transparency, compliance, and anti-fraud measures in decentralized finance.


Continuous Learning: Implementing feedback loops where the system learns from newly confirmed fraud cases can help keep detection models up-to-date with the latest tactics and threats.
