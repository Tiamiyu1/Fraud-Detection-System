import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING FRAUD DETECTION MODEL")
print("="*60)

# Load data
print("\n[1/6] Loading data...")
df = pd.read_csv('nova_pay_transactions.csv') 
print(f"✓ Loaded {len(df)} transactions")

# Data preparation
print("\n[2/6] Preparing data...")

# Fix data types
df['amount_src'] = pd.to_numeric(df['amount_src'].astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce')

# Handle missing values
# Numeric columns - fill with median
numeric_cols_to_fill = ['amount_usd', 'fee', 'device_trust_score']
for col in numeric_cols_to_fill:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

df['kyc_tier'] = df['kyc_tier'].str.strip().str.lower()

mapping_dict = {
    'standrd': 'standard',
    'enhancd': 'enhanced',
    'enh': 'enhanced',
    'std': 'standard',
    'unknown': None
}

df['kyc_tier'] = df['kyc_tier'].replace(mapping_dict).str.title()
df['kyc_tier'] = df['kyc_tier'].replace('Nan', None)
df['kyc_tier'] = df['kyc_tier'].fillna('Standard')
df['channel'] = df['channel'].str.strip().str.lower()

mapping_dict = {
    'mobille': 'mobile',
    'weeb': 'web',
    'unknown': None
}

df['channel'] = df['channel'].replace(mapping_dict).str.title()
df['channel'] = df['channel'].replace('Nan', None)
df['channel'] = df['channel'].fillna('Mobile')
df['channel'] = df['channel'].replace('Atm', 'ATM')
print(df.channel.unique())

df['ip_country'].fillna(df['home_country'], inplace=True)

# Parse timestamp
df['timestamp'].replace('0000-00-00T00:00:00Z', None)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['timestamp'].fillna(df['timestamp'].median(), inplace=True)

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

print("✓ Data cleaned and time features extracted")

# Feature engineering
print("\n[3/6] Engineering features...")

# Transaction behavior
df['high_velocity_flag'] = ((df['txn_velocity_1h'] >= 3) | (df['txn_velocity_24h'] >= 7)).astype(int)
df['new_account'] = (df['account_age_days'] < 50).astype(int)
df['old_account'] = (df['account_age_days'] > 365).astype(int)

# Amount features
df['amount_to_fee_ratio'] = np.where(df['fee'] > 0, df['amount_usd'] / df['fee'], 0)
df['amount_to_fee_ratio'] = df['amount_to_fee_ratio'].replace([np.inf, -np.inf], 0)
df['log_amount'] = np.log1p(df['amount_usd'])

# Amount bins
df['amount_category'] = pd.cut(df['amount_usd'], 
                                bins=[0, 100, 500, 1000, 5000, np.inf],
                                labels=['very_low', 'low', 'medium', 'high', 'very_high'])

df['risk_score_internal_normalized'] = df['risk_score_internal'] * 200
df['ip_risk_score_normalized'] = df['ip_risk_score'] * 83.33
df['corridor_risk_normalized'] = df['corridor_risk'] * 400
df['device_trust_score_normalized'] = df['device_trust_score'] * 100


# Risk composites
df['risk_composite'] = (
    df['ip_risk_score'] * 0.30 +
    df['risk_score_internal'] * 0.30 +
    df['corridor_risk'] * 0.20 +
    df['location_mismatch'].astype(int) * 20 * 0.20
)

df['device_risk'] = (
    (100 - df['device_trust_score']) * 0.6 +
    df['new_device'].astype(int) * 40 * 0.4
)

# Customer features
customer_avg_amount = df.groupby('customer_id')['amount_usd'].mean()
df['customer_avg_amount'] = df['customer_id'].map(customer_avg_amount)
customer_txn_count = df.groupby('customer_id').size()
df['customer_txn_count'] = df['customer_id'].map(customer_txn_count)
df['amount_deviation'] = np.abs(df['amount_usd'] - df['customer_avg_amount'])

# Interaction features
df['high_risk_new_device'] = ((df['risk_score_internal'] > 50) & df['new_device']).astype(int)
df['location_mismatch_velocity'] = df['location_mismatch'].astype(int) * df['high_velocity_flag']
df['night_high_amount'] = df['is_night'] * (df['amount_usd'] > df['amount_usd'].quantile(0.75)).astype(int)

print(f"✓ Created {21} engineered features")

# Select features
feature_columns = [
    # Risk scores (normalized)
    'risk_composite', 'device_risk',
    
    # Transaction patterns
    'txn_velocity_1h', 'txn_velocity_24h', 'high_velocity_flag',
    
    # Account characteristics
    'account_age_days', 'new_account', 'old_account',
    
    # Transaction details
    'amount_usd', 'log_amount', 'fee', 'amount_to_fee_ratio',
    
    # Behavioral flags
    'new_device', 'location_mismatch',
    
    # Time features
    'hour', 'day_of_week', 'is_weekend', 'is_night',
    
    # Customer patterns
    'customer_txn_count', 'amount_deviation',
    
    # Interaction features
    'high_risk_new_device', 'location_mismatch_velocity', 'night_high_amount'
]

# Convert boolean to int
bool_cols = df[feature_columns].select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

X = df[feature_columns].copy()
y = df['is_fraud'].copy()

# Train-test split
print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Handle class imbalance
print("\n[5/6] Applying SMOTE for class balance...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✓ Balanced training set: {len(X_train_balanced)} samples")

# Train model
print("\n[6/6] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save model and feature list
print("\n[SAVE] Saving model artifacts...")
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

# Save customer statistics for API
customer_stats = {
    'customer_avg_amount': customer_avg_amount.to_dict(),
    'customer_txn_count': customer_txn_count.to_dict()
}
joblib.dump(customer_stats, 'customer_stats.pkl')

print("✓ Saved: fraud_model.pkl")
print("✓ Saved: feature_columns.pkl")
print("✓ Saved: customer_stats.pkl")

print("\n" + "="*60)
print("✓ MODEL TRAINING COMPLETE!")
print("="*60)