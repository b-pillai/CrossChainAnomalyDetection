# Cross-Chain Anomaly Detection

## Overview

This project implements machine learning models to detect anomalous transactions in cross-chain bridge operations, specifically focusing on identifying potential Tornado Cash interactions. The system analyzes Circle's Cross-Chain Transfer Protocol (CCTP) bridge data to flag suspicious activities for compliance monitoring.

## 🎯 Objective

The primary goal is to develop robust supervised learning models that can:
- Detect transactions potentially linked to Tornado Cash
- Provide high recall rates to minimize false negatives in compliance scenarios
- Offer explainable AI insights for regulatory compliance
- Support real-time anomaly detection in cross-chain transactions

## 📊 Dataset

The analysis uses labeled CCTP bridge transaction data (`cctp_bridge_data_labeled.csv`) containing:
- **591,940 total transactions** across multiple blockchains
- **241 Tornado Cash labeled transactions (0.04% positive class)**
- Transaction metadata (hashes, timestamps, addresses)
- Cross-chain transfer details (source/destination chains, amounts)
- Compliance labels for supervised learning
- 12 raw features reduced to 7 engineered features for modeling

### Data Characteristics
- **Extreme Class Imbalance**: 0.04% positive cases (typical in fraud detection)
- **Multi-blockchain Coverage**: Arbitrum, Ethereum, and other chains
- **Temporal Range**: Multiple time periods for robust analysis
- **Clean Dataset**: No missing values, ready for analysis

**Note**: The dataset file is large (>50MB) and demonstrates real-world compliance detection challenges.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Conda or Miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/b-pillai/CrossChainAnomalyDetection.git
   cd CrossChainAnomalyDetection
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate cc-AD-1
   ```

3. **Verify installation**
   ```bash
   python -c "import pandas, sklearn, xgboost; print('All dependencies installed successfully!')"
   ```

### Usage

#### Basic Analysis
```bash
python SupervisedLearningCode_v1.py
```

#### Interactive Development
```bash
jupyter notebook
# Open SupervisedLearningCode_v1.py or create a new notebook
```

#### Programmatic Usage
```python
from SupervisedLearningCode_v1 import BridgeComplianceML

# Initialize analysis
ml_analysis = BridgeComplianceML('cctp_bridge_data_labeled.csv')

# Run complete pipeline
ml_analysis.run_complete_analysis()

# Optional: Hyperparameter tuning
best_model, best_params = ml_analysis.hyperparameter_tuning('Random Forest')
```

## 📁 Project Structure

```
CrossChainAnomalyDetection/
├── README.md                          # Project documentation
├── environment.yml                    # Conda environment specification
├── SupervisedLearningCode_v1.py      # Main ML analysis pipeline
├── cctp_bridge_data_labeled.csv      # Training dataset (591,940 transactions)
├── results.txt                       # Complete analysis results and metrics
├── LICENSE                           # MIT License
├── CONTRIBUTING.md                   # Contributor guidelines
├── requirements.txt                  # Alternative pip dependencies
├── .gitignore                        # Version control exclusions
└── generated_outputs/                # Visualizations created during analysis
    ├── model_performance_detailed.png # Model comparison chart
    ├── roc_curves.png                # ROC curve analysis  
    ├── precision_recall_curves.png   # Precision-Recall curves
    ├── top10_feature_importance.png  # Feature importance plot
    └── threshold_optimization.png    # Threshold tuning analysis
```

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Handling** - Zero-filling for numerical stability
2. **Feature Engineering** - Categorical encoding and scaling
3. **Feature Selection** - Statistical significance testing (f_classif)
4. **Data Splitting** - Stratified train/test split (80/20)

### Model Training
1. **Baseline Models** - Multiple algorithms for comparison
2. **Threshold Optimization** - Recall-focused for compliance scenarios
3. **Cross-Validation** - 5-fold CV for generalization assessment
4. **Hyperparameter Tuning** - Grid search for optimal parameters

### Evaluation Metrics
- **AUC-ROC** - Overall discriminative performance
- **Recall** - Minimizing false negatives (critical for compliance)
- **Precision** - Reducing false positives
- **F1-Score** - Harmonic mean of precision and recall

## 📈 Results

The analysis was conducted on **591,940 transactions** with **241 Tornado Cash matches (0.04% match rate)**, representing a highly imbalanced dataset typical of financial anomaly detection scenarios.

### Dataset Statistics
- **Total Transactions**: 591,940
- **Tornado Cash Matches**: 241 (0.04%)
- **Training Set**: 473,552 transactions
- **Test Set**: 118,388 transactions
- **Features Used**: 7 (src_blockchain, src_from_address, timestamps, amounts, etc.)

### Model Performance Summary
| Model | AUC | Recall | Precision | F1-Score | Threshold |
|-------|-----|--------|-----------|----------|-----------|
| **Random Forest** | **0.8293** | **0.3542** | **0.5000** | **0.4146** | 0.100 |
| **XGBoost** | **0.8756** | **0.2708** | **0.3824** | **0.3171** | 0.100 |
| Gradient Boosting | 0.8161 | 0.1875 | 0.1915 | 0.1895 | 0.500 |
| Logistic Regression | 0.7014 | 0.0000 | 0.0000 | 0.0000 | 0.500 |
| SVM | 0.4145 | 0.0000 | 0.0000 | 0.0000 | 0.500 |

### Key Findings

#### Best Performing Model: Random Forest
- **AUC Score**: 0.8293 (Strong discriminative performance)
- **Recall**: 35.42% (Detected 17 out of 48 suspicious transactions)
- **Precision**: 50.00% (Half of flagged transactions were true positives)
- **Cross-Validation AUC**: 0.8094 ± 0.0985

#### Hyperparameter Tuned Random Forest
After optimization, the Random Forest achieved:
- **Improved AUC**: 0.9065
- **Enhanced Precision**: 90.91% (much higher precision)
- **Moderate Recall**: 20.83% (10 out of 48 detected)
- **Optimal Parameters**: max_depth=10, n_estimators=200, min_samples_split=10

#### Top 5 Most Important Features
1. **src_from_address** (27.98%) - Source wallet address patterns
2. **dst_timestamp** (18.47%) - Destination transaction timing
3. **src_timestamp** (18.36%) - Source transaction timing  
4. **amount** (16.78%) - Transaction amount patterns
5. **amount_usd** (16.18%) - USD value patterns

### Performance Analysis

The results demonstrate the challenge of detecting rare anomalies in financial data:

- **XGBoost** achieved the highest AUC (0.8756) but with moderate recall
- **Random Forest** provided the best balance with threshold optimization
- **Traditional models** (Logistic Regression, SVM) struggled with the extreme class imbalance
- **Threshold optimization** was crucial for improving recall in compliance scenarios

## 📋 Dependencies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Disclaimer**: This tool is for research and compliance analysis purposes. Always verify results with domain experts and follow applicable regulations.
