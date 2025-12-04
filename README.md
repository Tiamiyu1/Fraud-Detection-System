# 🛡️ NOVA Pay Fraud Detection System

Real-time fraud detection for financial transactions using Machine Learning.

## 🚀 Quick Start

# Clone the repository to your local machine
# Navigate to the project folder

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train_model.py

# 3. Run the app
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

## 📦 Requirements

```txt
streamlit==1.28.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
imbalanced-learn==0.11.0
joblib==1.3.2
plotly==5.18.0
```

## 🎯 Features

- **Real-time fraud detection** with probability scores
- **Interactive UI** with pre-filled sample transactions
- **Visual analytics** including gauges and risk factors
- **Risk classification**: LOW/MEDIUM/HIGH/CRITICAL

## 📊 Model Performance

- Accuracy: ~93%
- Precision: ~90%
- Recall: ~88%
- ROC-AUC: ~0.96

## 🌐 Access on Streamlit Cloud

Go to [streamlit.io](https://uncoverfraud.streamlit.app/)



## 🎮 Usage

1. **Choose a sample** from the sidebar (Low Risk/High Risk) or enter custom values
2. **Fill in transaction details** using the form
3. **Click "Analyze Transaction"** to get instant fraud prediction
4. **Review results** including probability, risk level, and risk factors

## 🔧 Troubleshooting

**Model not found?**
```bash
python train_model.py
```

**Port already in use?**
```bash
streamlit run streamlit_app.py --server.port=8502
```

**Module not found?**
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
├── streamlit_app.py                # Main app
├── train_model.py                  # Model training
├── requirements.txt                # Dependencies
├── fraud_model.pkl                 # Trained model
├── feature_columns.pkl             # Features
├── customer_stats.pkl              # Customer data
├── nova_pay_fraud_detection.ipynb  # Jupyter Notebook (EDA, models, top features)
└── nova_pay_transactions.csv       # Dataset
```

## 🤖 How It Works

The system analyzes **28 features** including:
- Risk scores (internal, IP, device, corridor)
- Transaction patterns and velocity
- Account age and chargeback history
- Time-based features (hour, weekend, night)
- Behavioral flags (new device, location mismatch)

Uses **Random Forest Classifier** with SMOTE for class balancing.

## 📝 License

MIT License

---
