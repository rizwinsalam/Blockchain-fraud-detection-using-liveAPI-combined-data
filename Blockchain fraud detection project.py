import os
import sys
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Configure data source (forced combined): use both live API and CSV
DATA_MODE = 'combined'

# Configuration for sources
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', 'SKGRZ59HBF175JHG8XN4KARI12PIJ6PW71')
ETH_ADDRESS = os.getenv('ETH_ADDRESS', '0x742d35Cc6634C0532925a3b844Bc454e4438f44e')
CSV_PATH = os.getenv('CSV_PATH', r"C:\Users\USER\OneDrive - Berlin School of Business and Innovation (BSBI)\PYTHON\demo_fraud_dataset.csv")


def fetch_api_transactions(address: str, api_key: str) -> pd.DataFrame:
    url = (
        f"https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist&address={address}"
        f"&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"[API] Request failed: {exc}")
        return pd.DataFrame()

    if not isinstance(payload, dict) or payload.get('status') != '1':
        print(f"[API] Error or no data: {payload}")
        return pd.DataFrame()

    transactions = payload.get('result', [])
    if not transactions:
        return pd.DataFrame()
    return pd.DataFrame(transactions)


def normalize_api_df(df_api: pd.DataFrame) -> pd.DataFrame:
    if df_api.empty:
        return df_api

    cols_needed = ['from', 'to', 'value', 'gas', 'gasPrice', 'isError', 'txreceipt_status', 'timeStamp']
    missing = [c for c in cols_needed if c not in df_api.columns]
    if missing:
        # If essential columns are missing, return empty to avoid downstream errors
        print(f"[API] Missing columns, skipping API data: {missing}")
        return pd.DataFrame()

    df_api = df_api[cols_needed].copy()

    # Type conversions
    for col in ['value', 'gas', 'gasPrice']:
        df_api[col] = pd.to_numeric(df_api[col], errors='coerce')
    for col in ['isError', 'txreceipt_status']:
        df_api[col] = pd.to_numeric(df_api[col], errors='coerce').astype('Int64')
    df_api['timeStamp'] = pd.to_numeric(df_api['timeStamp'], errors='coerce')

    # Feature engineering
    df_api['value_eth'] = df_api['value'] / 1e18
    df_api['gas_fee_eth'] = (df_api['gas'] * df_api['gasPrice']) / 1e18
    df_api['timestamp'] = pd.to_datetime(df_api['timeStamp'], unit='s', errors='coerce')
    df_api['hour_of_day'] = df_api['timestamp'].dt.hour
    df_api['day_of_week'] = df_api['timestamp'].dt.dayofweek
    df_api['source'] = 'api'

    # Align to unified schema
    unified_cols = [
        'timestamp', 'from', 'to', 'value_eth', 'gas_fee_eth', 'hour_of_day', 'day_of_week',
        'isError', 'txreceipt_status', 'source'
    ]
    return df_api[unified_cols]


def load_csv_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[CSV] File not found: {csv_path}")
        return pd.DataFrame()
    try:
        df_csv = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[CSV] Load failed: {exc}")
        return pd.DataFrame()

    # Expect columns like: from, to, value_ETH, timestamp, optional is_fraud
    # Normalize names and types
    if 'value_ETH' in df_csv.columns:
        df_csv = df_csv.rename(columns={'value_ETH': 'value_eth'})
    if 'timestamp' in df_csv.columns:
        df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'], errors='coerce')

    # Synthesize fields to match API schema
    if 'value_eth' not in df_csv.columns and 'value' in df_csv.columns:
        # assume wei
        df_csv['value_eth'] = pd.to_numeric(df_csv['value'], errors='coerce') / 1e18

    df_csv['gas_fee_eth'] = pd.to_numeric(df_csv.get('gas_fee_eth', np.nan), errors='coerce')
    df_csv['hour_of_day'] = df_csv['timestamp'].dt.hour if 'timestamp' in df_csv.columns else np.nan
    df_csv['day_of_week'] = df_csv['timestamp'].dt.dayofweek if 'timestamp' in df_csv.columns else np.nan
    df_csv['isError'] = pd.Series(0, index=df_csv.index, dtype='Int64')  # not available, assume 0
    df_csv['txreceipt_status'] = pd.Series(1, index=df_csv.index, dtype='Int64')  # assume success
    df_csv['source'] = 'csv'

    # Ensure required columns exist
    for col in ['from', 'to', 'value_eth', 'timestamp']:
        if col not in df_csv.columns:
            df_csv[col] = np.nan

    unified_cols = [
        'timestamp', 'from', 'to', 'value_eth', 'gas_fee_eth', 'hour_of_day', 'day_of_week',
        'isError', 'txreceipt_status', 'source'
    ]
    return df_csv[unified_cols]


def build_unified_dataframe(mode: str) -> pd.DataFrame:
    mode = (mode or '').lower()
    use_api = mode in ('api', 'combined')
    use_csv = mode in ('csv', 'combined')

    frames = []

    if use_api:
        api_raw = fetch_api_transactions(ETH_ADDRESS, ETHERSCAN_API_KEY)
        api_df = normalize_api_df(api_raw)
        if not api_df.empty:
            frames.append(api_df)

    if use_csv:
        csv_df = load_csv_df(CSV_PATH)
        if not csv_df.empty:
            frames.append(csv_df)

    if not frames:
        print("[UNIFIED] No data available from selected sources.")
        return pd.DataFrame(columns=[
            'timestamp', 'from', 'to', 'value_eth', 'gas_fee_eth', 'hour_of_day', 'day_of_week',
            'isError', 'txreceipt_status', 'source'
        ])

    df_unified = pd.concat(frames, ignore_index=True)
    df_unified.sort_values('timestamp', inplace=True)
    df_unified.reset_index(drop=True, inplace=True)

    # Optional: fill missing gas_fee_eth with 0 for visualizations
    df_unified['gas_fee_eth'] = pd.to_numeric(df_unified['gas_fee_eth'], errors='coerce').fillna(0.0)

    return df_unified


# Build unified dataset once and expose canonical variables expected downstream
df = build_unified_dataframe(DATA_MODE)

# Derive df_clean consistent with previous code expectations
if not df.empty:
    df_clean = df.copy()
else:
    df_clean = df

# Persist a tidy table similar to previous workflow for reuse
df_table = df_clean.copy()
df_table = df_table.rename(columns={'from': 'sender', 'to': 'receiver', 'isError': 'error_flag', 'txreceipt_status': 'receipt_status'})

try:
    df_table.to_csv('eth_transactions_cleaned.csv', index=False)
except Exception as exc:
    print(f"[UNIFIED] Could not write CSV: {exc}")

# Quick preview
pd.set_option('display.max_columns', None)
print(f"[UNIFIED] Mode: {DATA_MODE} | Rows: {len(df)}")
print(df.head())

# ================================
# Unified analytics using df_clean
# ================================
if not df_clean.empty:
    # Prepare supervised target from API-style fields (fallback to 0 when missing)
    y_target = df_clean['isError'].fillna(0).astype(int)
    X_features = df_clean[['value_eth', 'gas_fee_eth', 'hour_of_day', 'day_of_week']].copy()

    # Train/test only if we have enough data and >1 class
    try:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        if len(df_clean) >= 50 and y_target.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_target, test_size=0.2, random_state=42
            )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

            lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
            print("[UNIFIED] Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
            print("[UNIFIED] LR Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
            print("[UNIFIED] LR Classification Report:\n", classification_report(y_test, y_pred_lr))

            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
            print("[UNIFIED] Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
            print("[UNIFIED] RF Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
            print("[UNIFIED] RF Classification Report:\n", classification_report(y_test, y_pred_rf))
        else:
            print("[UNIFIED] Skipping supervised training (insufficient rows or single class).")
    except Exception as exc:
        print(f"[UNIFIED] Supervised section skipped due to error: {exc}")

    # Unsupervised clustering for anomaly sense-making
    try:
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler as _StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

        X_cluster = df_clean[['value_eth', 'gas_fee_eth', 'hour_of_day', 'day_of_week']].copy()
        scaler_c = _StandardScaler()
        X_scaled = scaler_c.fit_transform(X_cluster)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)
df_clean['cluster'] = clusters

plt.figure(figsize=(8,6))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set1', s=80)
        plt.title("PCA + KMeans Clustering (Unified)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
plt.show()

        cluster_stats = df_clean.groupby('cluster')[['value_eth','gas_fee_eth','hour_of_day','day_of_week']].mean()
        print("[UNIFIED] Cluster means:\n", cluster_stats)

anomaly_cluster = cluster_stats['value_eth'].idxmax()
        df_clean['is_anomaly'] = (df_clean['cluster'] == anomaly_cluster).astype(int)
        print("[UNIFIED] Anomaly counts:\n", df_clean['is_anomaly'].value_counts())

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_clean['is_anomaly'], palette={0:'blue',1:'red'}, s=80)
        plt.title("Anomaly Tagging (Unified)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
plt.show()
    except Exception as exc:
        print(f"[UNIFIED] Unsupervised section skipped due to error: {exc}")

    # ================================
    # Showcase charts (all based on df_clean)
    # ================================
    try:
        # Ensure timestamps are valid
        df_ts = df_clean.dropna(subset=['timestamp']).copy()

        # Daily transaction count
        daily_count = df_ts.groupby(df_ts['timestamp'].dt.date).size()
plt.figure(figsize=(12,5))
        daily_count.plot(kind='line', marker='o', color='#1f77b4')
        plt.title('Daily Transaction Count (Unified)')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
plt.show()

        # Monthly transaction count
        monthly_count = df_ts.groupby(df_ts['timestamp'].dt.to_period('M')).size()
plt.figure(figsize=(12,5))
        monthly_count.index = monthly_count.index.astype(str)
        sns.barplot(x=monthly_count.index, y=monthly_count.values, color='#5DADE2')
        plt.title('Monthly Transaction Count (Unified)')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

        # Top 10 senders/receivers by total ETH value
        top_senders = (
            df_clean.groupby('from', dropna=True)['value_eth']
            .sum().sort_values(ascending=False).head(10).reset_index()
        )
plt.figure(figsize=(10,6))
        sns.barplot(data=top_senders, x='value_eth', y='from', color='#58D68D')
        plt.title('Top 10 Senders by Total ETH Sent (Unified)')
plt.xlabel('Total ETH Sent')
plt.ylabel('Sender Address')
        plt.tight_layout()
plt.show()

        top_receivers = (
            df_clean.groupby('to', dropna=True)['value_eth']
            .sum().sort_values(ascending=False).head(10).reset_index()
        )
plt.figure(figsize=(10,6))
        sns.barplot(data=top_receivers, x='value_eth', y='to', color='#F5B041')
        plt.title('Top 10 Receivers by Total ETH Received (Unified)')
plt.xlabel('Total ETH Received')
plt.ylabel('Receiver Address')
        plt.tight_layout()
plt.show()

        # Value distribution (hist + optional log y)
plt.figure(figsize=(10,6))
        sns.histplot(df_clean['value_eth'].dropna(), bins=60, color='#8E44AD', alpha=0.8)
        plt.title('Distribution of Transaction Values (ETH)')
plt.xlabel('Transaction Value (ETH)')
plt.ylabel('Frequency')
plt.yscale('log')
        plt.tight_layout()
plt.show()

        # Boxplot for outliers
        plt.figure(figsize=(10,5))
        sns.boxplot(x=df_clean['value_eth'].dropna(), color='#AF7AC5')
        plt.title('Transaction Values and Outliers (ETH)')
        plt.tight_layout()
plt.show()

        # Correlation heatmap
        corr_cols = ['value_eth', 'gas_fee_eth']
        corr_df = df_clean[corr_cols].copy()
        corr = corr_df.corr()
        plt.figure(figsize=(5,4))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
plt.show()

        # Scatter: Value vs Gas Fee
plt.figure(figsize=(10,6))
        sns.scatterplot(data=df_clean, x='value_eth', y='gas_fee_eth', hue='source', alpha=0.7)
        plt.title('Transaction Value vs Gas Fee (Unified, colored by source)')
plt.xlabel('Transaction Value (ETH)')
        plt.ylabel('Gas Fee (ETH)')
        plt.tight_layout()
plt.show()
    except Exception as exc:
        print(f"[UNIFIED] Charts section skipped due to error: {exc}")

    # ================================
    # Normal vs Fraud/Anomaly charts
    # ================================
    try:
        # Prefer model-agnostic anomaly flag; fallback to isError
        if 'is_anomaly' in df_clean.columns:
            fraud_flag = df_clean['is_anomaly'].fillna(0).astype(int)
            label_title = 'Normal vs Anomaly (Unified)'
            legend_labels = {0: 'Normal', 1: 'Anomaly'}
        elif 'isError' in df_clean.columns:
            fraud_flag = df_clean['isError'].fillna(0).astype(int)
            label_title = 'Normal vs Error (Unified)'
            legend_labels = {0: 'Normal', 1: 'Error'}
        else:
            fraud_flag = pd.Series(0, index=df_clean.index)
            label_title = 'Normal vs Fraud (Unified)'
            legend_labels = {0: 'Normal', 1: 'Fraud'}

        # Count bar chart
        counts = fraud_flag.value_counts().reindex([0,1], fill_value=0)
        plt.figure(figsize=(6,4))
        sns.barplot(x=[legend_labels.get(0,'Normal'), legend_labels.get(1,'Fraud')], y=counts.values, palette=['#2ecc71','#e74c3c'])
        plt.title(f'{label_title} - Count')
        plt.ylabel('Number of Transactions')
        plt.xlabel('Class')
        plt.tight_layout()
        plt.show()

        # Value share pie chart
        value_share = df_clean.groupby(fraud_flag)['value_eth'].sum().reindex([0,1], fill_value=0)
        plt.figure(figsize=(6,6))
        plt.pie(value_share.values, labels=[legend_labels.get(0,'Normal'), legend_labels.get(1,'Fraud')], autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'], startangle=140)
        plt.title(f'{label_title} - Share of Total ETH')
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"[UNIFIED] Normal vs Fraud charts skipped due to error: {exc}")

# Stop before legacy duplicated cells
sys.exit(0)
