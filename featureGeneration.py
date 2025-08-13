import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CCTPFeatureGenerator:
    def __init__(self, data_file):
        """
        Initialize the feature generator for CCTP bridge data
        """
        self.data_file = data_file
        self.df = None
        self.features_generated = []
        
    def load_data(self):
        """
        Load the CCTP bridge dataset
        """
        print("Loading CCTP bridge dataset...")
        self.df = pd.read_csv(self.data_file)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self.df
    
    def generate_amount_based_features(self):
        """
        Generate features based on transaction amounts and USD values
        """
        print("\nGenerating amount-based features...")
        
        # 1. Amount percentile ranking
        self.df['amount_usd_percentile'] = self.df['amount_usd'].rank(pct=True)
        
        # 2. Amount category based on percentiles
        def categorize_amount(percentile):
            if percentile <= 0.25:
                return 'very_low'
            elif percentile <= 0.50:
                return 'low'
            elif percentile <= 0.75:
                return 'medium'
            elif percentile <= 0.90:
                return 'high'
            else:
                return 'very_high'
        
        self.df['amount_category'] = self.df['amount_usd_percentile'].apply(categorize_amount)
        
        # 3. Binary flags for extreme amounts
        self.df['is_very_low_amount'] = (self.df['amount_usd_percentile'] <= 0.10).astype(int)
        self.df['is_low_amount'] = (self.df['amount_usd_percentile'] <= 0.25).astype(int)
        self.df['is_high_amount'] = (self.df['amount_usd_percentile'] >= 0.75).astype(int)
        self.df['is_very_high_amount'] = (self.df['amount_usd_percentile'] >= 0.90).astype(int)
        
        # 4. Round number detection (common in suspicious activities)
        self.df['is_round_amount'] = (self.df['amount'] % 1000 == 0).astype(int)
        self.df['is_round_usd'] = (self.df['amount_usd'] % 100 == 0).astype(int)
        
        # 5. Amount deviation from address mean
        address_stats = self.df.groupby('depositor')['amount_usd'].agg(['mean', 'std', 'count']).reset_index()
        address_stats.columns = ['depositor', 'depositor_avg_amount', 'depositor_std_amount', 'depositor_tx_count']
        
        self.df = self.df.merge(address_stats, on='depositor', how='left')
        
        # Calculate z-score for amount deviation
        self.df['amount_zscore'] = np.where(
            self.df['depositor_std_amount'] > 0,
            (self.df['amount_usd'] - self.df['depositor_avg_amount']) / self.df['depositor_std_amount'],
            0
        )
        
        self.features_generated.extend([
            'amount_usd_percentile', 'amount_category', 'is_very_low_amount', 
            'is_low_amount', 'is_high_amount', 'is_very_high_amount',
            'is_round_amount', 'is_round_usd', 'depositor_avg_amount',
            'depositor_std_amount', 'depositor_tx_count', 'amount_zscore'
        ])
        
        print(f"Generated {len(self.features_generated)} amount-based features")
        
    def generate_address_pattern_features(self):
        """
        Generate features based on address transaction patterns
        """
        print("\nGenerating address pattern features...")
        
        # 1. Transaction frequency patterns
        depositor_patterns = self.df.groupby('depositor').agg({
            'amount_usd': ['count', 'sum', 'mean', 'median', 'std', 'min', 'max'],
            'src_blockchain': lambda x: x.nunique(),
            'dst_blockchain': lambda x: x.nunique(),
            'recipient': lambda x: x.nunique()
        }).reset_index()
        
        # Flatten column names
        depositor_patterns.columns = [
            'depositor', 'tx_count', 'total_volume', 'avg_amount', 'median_amount', 
            'std_amount', 'min_amount', 'max_amount', 'unique_src_chains', 
            'unique_dst_chains', 'unique_recipients'
        ]
        
        # 2. Detect frequent small transactions (potential structuring)
        low_threshold = self.df['amount_usd'].quantile(0.25)
        
        small_tx_patterns = self.df[self.df['amount_usd'] <= low_threshold].groupby('depositor').agg({
            'amount_usd': 'count'
        }).reset_index()
        small_tx_patterns.columns = ['depositor', 'small_tx_count']
        
        depositor_patterns = depositor_patterns.merge(small_tx_patterns, on='depositor', how='left')
        depositor_patterns['small_tx_count'] = depositor_patterns['small_tx_count'].fillna(0)
        
        # 3. Calculate ratios and patterns
        depositor_patterns['small_tx_ratio'] = depositor_patterns['small_tx_count'] / depositor_patterns['tx_count']
        depositor_patterns['amount_range'] = depositor_patterns['max_amount'] - depositor_patterns['min_amount']
        depositor_patterns['coefficient_variation'] = depositor_patterns['std_amount'] / depositor_patterns['avg_amount']
        depositor_patterns['coefficient_variation'] = depositor_patterns['coefficient_variation'].fillna(0)
        
        # 4. Structuring detection flags
        depositor_patterns['potential_structuring'] = (
            (depositor_patterns['small_tx_count'] >= 5) & 
            (depositor_patterns['small_tx_ratio'] >= 0.7)
        ).astype(int)
        
        depositor_patterns['high_frequency_small'] = (
            (depositor_patterns['small_tx_count'] >= 10) & 
            (depositor_patterns['avg_amount'] <= low_threshold * 1.5)
        ).astype(int)
        
        # 5. Chain diversity patterns
        depositor_patterns['is_cross_chain_frequent'] = (
            depositor_patterns['unique_src_chains'] >= 2
        ).astype(int)
        
        depositor_patterns['is_multi_recipient'] = (
            depositor_patterns['unique_recipients'] >= 3
        ).astype(int)
        
        # Merge back to main dataframe
        pattern_features = [
            'tx_count', 'total_volume', 'avg_amount', 'median_amount', 'std_amount',
            'min_amount', 'max_amount', 'unique_src_chains', 'unique_dst_chains',
            'unique_recipients', 'small_tx_count', 'small_tx_ratio', 'amount_range',
            'coefficient_variation', 'potential_structuring', 'high_frequency_small',
            'is_cross_chain_frequent', 'is_multi_recipient'
        ]
        
        # Rename columns to avoid conflicts
        pattern_features_renamed = ['addr_' + col for col in pattern_features]
        depositor_patterns.columns = ['depositor'] + pattern_features_renamed
        
        self.df = self.df.merge(depositor_patterns, on='depositor', how='left')
        
        self.features_generated.extend(pattern_features_renamed)
        print(f"Generated {len(pattern_features_renamed)} address pattern features")
        
    def generate_temporal_features(self):
        """
        Generate temporal-based features
        """
        print("\nGenerating temporal features...")
        
        # Convert timestamps if they're not already datetime
        if 'src_timestamp' in self.df.columns:
            if self.df['src_timestamp'].dtype != 'datetime64[ns]':
                self.df['src_timestamp_dt'] = pd.to_datetime(self.df['src_timestamp'], unit='s')
            else:
                self.df['src_timestamp_dt'] = self.df['src_timestamp']
        
        if 'dst_timestamp' in self.df.columns:
            if self.df['dst_timestamp'].dtype != 'datetime64[ns]':
                self.df['dst_timestamp_dt'] = pd.to_datetime(self.df['dst_timestamp'], unit='s')
            else:
                self.df['dst_timestamp_dt'] = self.df['dst_timestamp']
        
        # Extract temporal components
        self.df['src_hour'] = self.df['src_timestamp_dt'].dt.hour
        self.df['src_day_of_week'] = self.df['src_timestamp_dt'].dt.dayofweek
        self.df['src_is_weekend'] = (self.df['src_day_of_week'].isin([5, 6])).astype(int)
        self.df['src_is_night'] = ((self.df['src_hour'] >= 22) | (self.df['src_hour'] <= 6)).astype(int)
        
        # Time difference between src and dst
        if 'dst_timestamp_dt' in self.df.columns:
            self.df['bridge_time_diff'] = (self.df['dst_timestamp_dt'] - self.df['src_timestamp_dt']).dt.total_seconds()
            self.df['bridge_time_hours'] = self.df['bridge_time_diff'] / 3600
        
        # Sort by timestamp for time-based features
        self.df = self.df.sort_values('src_timestamp_dt').reset_index(drop=True)
        
        # Time-based address patterns
        address_temporal = self.df.groupby('depositor')['src_timestamp_dt'].agg([
            lambda x: (x.max() - x.min()).total_seconds() / 86400,  # Activity span in days
            lambda x: len(x.dt.hour.unique()),  # Unique hours of activity
            lambda x: len(x.dt.date.unique())   # Unique days of activity
        ]).reset_index()
        
        address_temporal.columns = ['depositor', 'activity_span_days', 'unique_hours', 'unique_days']
        
        # Activity concentration
        address_temporal['activity_concentration'] = address_temporal['unique_hours'] / 24
        address_temporal['daily_frequency'] = self.df.groupby('depositor').size().values / address_temporal['activity_span_days']
        address_temporal['daily_frequency'] = address_temporal['daily_frequency'].replace([np.inf, -np.inf], 0)
        
        self.df = self.df.merge(address_temporal, on='depositor', how='left')
        
        temporal_features = [
            'src_hour', 'src_day_of_week', 'src_is_weekend', 'src_is_night',
            'activity_span_days', 'unique_hours', 'unique_days', 
            'activity_concentration', 'daily_frequency'
        ]
        
        if 'bridge_time_diff' in self.df.columns:
            temporal_features.extend(['bridge_time_diff', 'bridge_time_hours'])
        
        self.features_generated.extend(temporal_features)
        print(f"Generated {len(temporal_features)} temporal features")
        
    def generate_risk_scores(self):
        """
        Generate composite risk scores based on multiple factors
        """
        print("\nGenerating composite risk scores...")
        
        # 1. Structuring Risk Score
        structuring_score = 0
        
        # High number of small transactions
        if 'addr_small_tx_count' in self.df.columns:
            structuring_score += np.where(self.df['addr_small_tx_count'] >= 10, 3, 0)
            structuring_score += np.where(self.df['addr_small_tx_count'] >= 5, 2, 0)
        
        # High ratio of small transactions
        if 'addr_small_tx_ratio' in self.df.columns:
            structuring_score += np.where(self.df['addr_small_tx_ratio'] >= 0.8, 3, 0)
            structuring_score += np.where(self.df['addr_small_tx_ratio'] >= 0.6, 2, 0)
        
        # Round amounts (common in structuring)
        structuring_score += self.df['is_round_amount'] * 1
        structuring_score += self.df['is_round_usd'] * 1
        
        self.df['structuring_risk_score'] = structuring_score
        
        # 2. Velocity Risk Score
        velocity_score = 0
        
        if 'daily_frequency' in self.df.columns:
            velocity_score += np.where(self.df['daily_frequency'] >= 10, 3, 0)
            velocity_score += np.where(self.df['daily_frequency'] >= 5, 2, 0)
        
        if 'addr_tx_count' in self.df.columns:
            velocity_score += np.where(self.df['addr_tx_count'] >= 20, 2, 0)
            velocity_score += np.where(self.df['addr_tx_count'] >= 10, 1, 0)
        
        self.df['velocity_risk_score'] = velocity_score
        
        # 3. Diversity Risk Score (high diversity can indicate mixing)
        diversity_score = 0
        
        if 'addr_unique_recipients' in self.df.columns:
            diversity_score += np.where(self.df['addr_unique_recipients'] >= 5, 2, 0)
            diversity_score += np.where(self.df['addr_unique_recipients'] >= 3, 1, 0)
        
        if 'addr_unique_src_chains' in self.df.columns:
            diversity_score += np.where(self.df['addr_unique_src_chains'] >= 3, 2, 0)
        
        if 'addr_is_cross_chain_frequent' in self.df.columns:
            diversity_score += self.df['addr_is_cross_chain_frequent'] * 1
        
        self.df['diversity_risk_score'] = diversity_score
        
        # 4. Temporal Risk Score
        temporal_score = 0
        
        # Night time activity
        temporal_score += self.df['src_is_night'] * 1
        
        # Weekend activity
        temporal_score += self.df['src_is_weekend'] * 1
        
        # High activity concentration (automated behavior)
        if 'activity_concentration' in self.df.columns:
            temporal_score += np.where(self.df['activity_concentration'] <= 0.2, 2, 0)  # Active in few hours
        
        self.df['temporal_risk_score'] = temporal_score
        
        # 5. Combined Risk Score
        self.df['combined_risk_score'] = (
            self.df['structuring_risk_score'] + 
            self.df['velocity_risk_score'] + 
            self.df['diversity_risk_score'] + 
            self.df['temporal_risk_score']
        )
        
        # Normalize to 0-10 scale
        max_score = self.df['combined_risk_score'].max()
        if max_score > 0:
            self.df['normalized_risk_score'] = (self.df['combined_risk_score'] / max_score) * 10
        else:
            self.df['normalized_risk_score'] = 0
        
        # Risk categories
        def categorize_risk(score):
            if score <= 2:
                return 'low'
            elif score <= 5:
                return 'medium'
            elif score <= 7:
                return 'high'
            else:
                return 'very_high'
        
        self.df['risk_category'] = self.df['normalized_risk_score'].apply(categorize_risk)
        
        risk_features = [
            'structuring_risk_score', 'velocity_risk_score', 'diversity_risk_score',
            'temporal_risk_score', 'combined_risk_score', 'normalized_risk_score',
            'risk_category'
        ]
        
        self.features_generated.extend(risk_features)
        print(f"Generated {len(risk_features)} risk score features")
        
    def generate_all_features(self):
        """
        Generate all feature sets
        """
        print("Starting comprehensive feature generation...")
        
        # Load data
        self.load_data()
        
        # Generate all feature types
        self.generate_amount_based_features()
        self.generate_address_pattern_features()
        self.generate_temporal_features()
        self.generate_risk_scores()
        
        print(f"\nFeature generation complete!")
        print(f"Total features generated: {len(self.features_generated)}")
        print(f"Original dataset shape: {self.df.shape}")
        
        return self.df
    
    def save_enhanced_dataset(self, output_file=None):
        """
        Save the dataset with new features
        """
        if output_file is None:
            output_file = self.data_file.replace('.csv', '_enhanced.csv')
        
        print(f"\nSaving enhanced dataset to: {output_file}")
        self.df.to_csv(output_file, index=False)
        
        # Generate feature summary
        feature_summary = pd.DataFrame({
            'feature': self.features_generated,
            'type': ['generated'] * len(self.features_generated),
            'description': self._get_feature_descriptions()
        })
        
        summary_file = output_file.replace('.csv', '_feature_summary.csv')
        feature_summary.to_csv(summary_file, index=False)
        print(f"Feature summary saved to: {summary_file}")
        
        return output_file
    
    def _get_feature_descriptions(self):
        """
        Get descriptions for generated features
        """
        descriptions = {
            'amount_usd_percentile': 'Percentile rank of transaction amount in USD',
            'amount_category': 'Categorical amount level (very_low to very_high)',
            'is_very_low_amount': 'Binary flag for very low amounts (<=10th percentile)',
            'is_low_amount': 'Binary flag for low amounts (<=25th percentile)',
            'is_high_amount': 'Binary flag for high amounts (>=75th percentile)',
            'is_very_high_amount': 'Binary flag for very high amounts (>=90th percentile)',
            'is_round_amount': 'Binary flag for round amounts (divisible by 1000)',
            'is_round_usd': 'Binary flag for round USD amounts (divisible by 100)',
            'depositor_avg_amount': 'Average transaction amount for depositor address',
            'depositor_std_amount': 'Standard deviation of amounts for depositor',
            'depositor_tx_count': 'Total transaction count for depositor',
            'amount_zscore': 'Z-score of amount relative to depositor average',
            'addr_tx_count': 'Total transactions by address',
            'addr_total_volume': 'Total volume transacted by address',
            'addr_small_tx_count': 'Count of small transactions by address',
            'addr_small_tx_ratio': 'Ratio of small to total transactions',
            'addr_potential_structuring': 'Binary flag for potential structuring pattern',
            'addr_high_frequency_small': 'Binary flag for high frequency small transactions',
            'structuring_risk_score': 'Composite score for structuring behavior',
            'velocity_risk_score': 'Composite score for transaction velocity',
            'diversity_risk_score': 'Composite score for transaction diversity',
            'temporal_risk_score': 'Composite score for temporal patterns',
            'combined_risk_score': 'Overall risk score combining all factors',
            'normalized_risk_score': 'Risk score normalized to 0-10 scale',
            'risk_category': 'Risk category (low, medium, high, very_high)'
        }
        
        return [descriptions.get(feature, 'Generated feature') for feature in self.features_generated]
    
    def analyze_suspicious_patterns(self):
        """
        Analyze and report on suspicious patterns detected
        """
        print("\n" + "="*60)
        print("SUSPICIOUS PATTERN ANALYSIS")
        print("="*60)
        
        # High-risk transactions
        high_risk = self.df[self.df['risk_category'].isin(['high', 'very_high'])]
        print(f"\nHigh-risk transactions detected: {len(high_risk)} ({len(high_risk)/len(self.df)*100:.2f}%)")
        
        # Potential structuring
        if 'addr_potential_structuring' in self.df.columns:
            structuring = self.df[self.df['addr_potential_structuring'] == 1]
            print(f"Potential structuring addresses: {structuring['depositor'].nunique()}")
            print(f"Transactions from structuring addresses: {len(structuring)}")
        
        # High frequency patterns
        if 'addr_high_frequency_small' in self.df.columns:
            high_freq = self.df[self.df['addr_high_frequency_small'] == 1]
            print(f"High frequency small transaction addresses: {high_freq['depositor'].nunique()}")
        
        # Risk score distribution
        print(f"\nRisk Category Distribution:")
        risk_dist = self.df['risk_category'].value_counts()
        for category, count in risk_dist.items():
            print(f"  {category}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Top risky addresses
        print(f"\nTop 10 Riskiest Addresses:")
        top_risky = self.df.nlargest(10, 'normalized_risk_score')[
            ['depositor', 'normalized_risk_score', 'risk_category', 'addr_tx_count', 'addr_total_volume']
        ]
        print(top_risky.to_string(index=False))


def main():
    """
    Main function to run feature generation
    """
    # Initialize feature generator
    # Update this path to your actual dataset file
    data_file = 'cctp_bridge_data_labeled.csv'  # Update with your actual filename
    
    generator = CCTPFeatureGenerator(data_file)
    
    # Generate all features
    enhanced_df = generator.generate_all_features()
    
    # Save enhanced dataset
    output_file = generator.save_enhanced_dataset()
    
    # Analyze suspicious patterns
    generator.analyze_suspicious_patterns()
    
    print(f"\nEnhanced dataset saved as: {output_file}")
    print("Feature generation complete!")


if __name__ == "__main__":
    main()
