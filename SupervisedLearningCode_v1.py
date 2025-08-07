import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class BridgeComplianceML:
    def __init__(self, data_file):
        """
        Initialize the machine learning analysis for bridge compliance data
        """
        self.data_file = data_file
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.optimized_thresholds = {}
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("Loading and exploring the dataset...")
        self.df = pd.read_csv(self.data_file)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        print(f"\nLabel distribution:")
        label_counts = self.df['tornado_cash_label'].value_counts()
        print(label_counts)
        print(f"Percentage of Tornado Cash matches: {label_counts[1]/len(self.df)*100:.2f}%")
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        print(f"\nData types:")
        print(self.df.dtypes)
        
        return self.df
    
    def preprocess_data(self):
        """
        Preprocess the data for machine learning
        """
        print("\nPreprocessing data...")
        
        # Make a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle missing values
        df_processed = df_processed.fillna(0)
        
        # Features to exclude from modeling
        exclude_columns = [
            'tornado_cash_label',  # This is our target
            'depositor',           # Text address
            'recipient',           # Text address
            'src_transaction_hash', # Unique identifier
            'dst_transaction_hash', # Unique identifier
            'timestamp',           # Will be processed separately if needed
        ]
        
        # Select numeric and categorical features
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
        
        # Encode categorical variables
        label_encoders = {}
        for col in feature_columns:
            if df_processed[col].dtype == 'object':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Prepare features and target
        self.X = df_processed[feature_columns]
        self.y = df_processed['tornado_cash_label']
        
        print(f"Features selected: {list(self.X.columns)}")
        print(f"Feature matrix shape: {self.X.shape}")
        
        return self.X, self.y
    
    def split_and_scale_data(self, test_size=0.2, random_state=42):
        """
        Split data into train/test sets and apply scaling
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set label distribution:")
        print(self.y_train.value_counts())
        
    def feature_selection(self, k=10):
        """
        Perform feature selection to identify most important features
        """
        print(f"\nPerforming feature selection (top {k} features)...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)
        
        # Get selected feature names
        selected_features = self.X.columns[selector.get_support()].tolist()
        print(f"Selected features: {selected_features}")
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': self.X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        print(f"\nTop 10 feature scores:")
        print(feature_scores.head(10))
        
        return X_train_selected, X_test_selected, selected_features, feature_scores
    
    def optimize_threshold_for_recall(self, model_name, y_true, y_prob, target_recall=0.85):
        """
        Find the optimal threshold that maximizes recall while maintaining reasonable precision
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        best_recall = 0
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:  # Check if both classes are predicted
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                results.append({
                    'threshold': threshold,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1
                })
                
                # Prioritize recall over precision with minimum acceptable precision
                if recall >= target_recall and precision >= 0.1:  # Minimum 10% precision
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                        best_recall = recall
        
        # If no threshold meets target recall, find the one with highest recall
        if best_threshold == 0.5 and results:
            best_result = max(results, key=lambda x: x['recall'])
            best_threshold = best_result['threshold']
            best_recall = best_result['recall']
        
        print(f"{model_name} - Optimal threshold: {best_threshold:.3f}, Recall: {best_recall:.3f}")
        return best_threshold
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("\nTraining multiple machine learning models...")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train on scaled data
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Optimize threshold for recall-focused models
            if name in ['Random Forest', 'XGBoost']:
                optimal_threshold = self.optimize_threshold_for_recall(name, self.y_test, y_pred_proba)
                self.optimized_thresholds[name] = optimal_threshold
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            else:
                y_pred = model.predict(self.X_test_scaled)
                self.optimized_thresholds[name] = 0.5
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            recall = recall_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score,
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'threshold': self.optimized_thresholds[name]
            }
            
            print(f"{name} - AUC: {auc_score:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
            print(f"        Threshold: {self.optimized_thresholds[name]:.3f}")
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for the best performing model
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingClassifier(random_state=42)
        
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate best model
        best_model = grid_search.best_estimator_
        y_pred_proba = best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Apply optimized threshold if it's a recall-focused model
        if model_name in ['Random Forest', 'XGBoost']:
            optimal_threshold = self.optimize_threshold_for_recall(model_name, self.y_test, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        else:
            y_pred = best_model.predict(self.X_test_scaled)
        
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        
        print(f"Test AUC with best model: {auc_score:.4f}")
        print(f"Test Recall: {recall:.4f}, Precision: {precision:.4f}")
        
        return best_model, grid_search.best_params_
    
    def evaluate_models(self):
        """
        Evaluate and compare all trained models
        """
        print("\nModel Evaluation Summary:")
        print("=" * 80)
        
        for name, results in self.results.items():
            print(f"\n{name}:")
            print(f"  Threshold: {results['threshold']:.3f}")
            print(f"  AUC Score: {results['auc_score']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  F1 Score: {results['f1_score']:.4f}")
            print(f"  CV AUC: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
            
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, results['predictions']))
            
            print(f"Confusion Matrix:")
            print(confusion_matrix(self.y_test, results['predictions']))
    
    def plot_results(self):
        """
        Create and save separate visualizations for model performance, ROC curves, and feature importance.
        """
        print("\nCreating visualizations...")

        # 1. Model Performance Comparison (including recall)
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        recall_scores = [self.results[name]['recall'] for name in model_names]
        precision_scores = [self.results[name]['precision'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, auc_scores, width, label='AUC', alpha=0.7)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.7)
        plt.bar(x + width, precision_scores, width, label='Precision', alpha=0.7)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison (AUC, Recall, Precision)')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_performance_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. ROC Curves
        plt.figure(figsize=(8, 6))
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {results['auc_score']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Precision-Recall Curves
        plt.figure(figsize=(8, 6))
        for name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, results['probabilities'])
            plt.plot(recall, precision, label=f"{name} (F1 = {results['f1_score']:.3f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Feature importance (for Random Forest)
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)

            plt.figure(figsize=(8, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title('Top 10 Feature Importance (Random Forest)')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('top10_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Threshold vs Metrics for recall-optimized models
        if 'Random Forest' in self.models or 'XGBoost' in self.models:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            for idx, model_name in enumerate(['Random Forest', 'XGBoost']):
                if model_name in self.models:
                    y_prob = self.results[model_name]['probabilities']
                    thresholds = np.arange(0.1, 0.9, 0.01)
                    recalls, precisions, f1s = [], [], []
                    
                    for threshold in thresholds:
                        y_pred = (y_prob >= threshold).astype(int)
                        if len(np.unique(y_pred)) > 1:
                            recalls.append(recall_score(self.y_test, y_pred))
                            precisions.append(precision_score(self.y_test, y_pred))
                            f1s.append(f1_score(self.y_test, y_pred))
                        else:
                            recalls.append(0)
                            precisions.append(0)
                            f1s.append(0)
                    
                    ax = axes[idx] if 'XGBoost' in self.models else axes
                    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
                    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
                    ax.plot(thresholds, f1s, label='F1-Score', linewidth=2)
                    ax.axvline(self.results[model_name]['threshold'], color='red', linestyle='--', 
                              label=f'Optimal Threshold ({self.results[model_name]["threshold"]:.3f})')
                    ax.set_xlabel('Threshold')
                    ax.set_ylabel('Score')
                    ax.set_title(f'{model_name} - Threshold vs Metrics')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self):
        """
        Generate a comprehensive analysis report
        """
        print("\n" + "="*60)
        print("BRIDGE COMPLIANCE ANALYSIS - TORNADO CASH DETECTION")
        print("="*60)
        
        print(f"\nDataset: {self.data_file}")
        print(f"Total transactions: {len(self.df)}")
        print(f"Tornado Cash matches: {self.df['tornado_cash_label'].sum()}")
        print(f"Match rate: {self.df['tornado_cash_label'].mean()*100:.2f}%")
        
        print(f"\nBest performing model (by recall):")
        best_model = max(self.results.items(), key=lambda x: x[1]['recall'])
        print(f"Model: {best_model[0]}")
        print(f"Recall: {best_model[1]['recall']:.4f}")
        print(f"Precision: {best_model[1]['precision']:.4f}")
        print(f"F1 Score: {best_model[1]['f1_score']:.4f}")
        print(f"AUC Score: {best_model[1]['auc_score']:.4f}")
        print(f"Threshold: {best_model[1]['threshold']:.3f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in self.models:
            print(f"\nTop 5 Most Important Features:")
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
                print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    def run_complete_analysis(self):
        """
        Run the complete machine learning analysis pipeline
        """
        print("Starting complete Bridge Compliance ML Analysis...")
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Split and scale data
        self.split_and_scale_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Create visualizations
        self.plot_results()
        
        # Generate report
        self.generate_report()
        
        print("\nAnalysis complete!")


def main():
    """
    Main function to run the analysis
    """
    # Initialize the ML analysis
    ml_analysis = BridgeComplianceML('cctp_bridge_data_labeled.csv')
    
    # Run complete analysis
    ml_analysis.run_complete_analysis()
    
    # Optional: Perform hyperparameter tuning on best model
    print("\nWould you like to perform hyperparameter tuning? (This may take some time)")
    best_model, best_params = ml_analysis.hyperparameter_tuning('Random Forest')


if __name__ == "__main__":
    main()