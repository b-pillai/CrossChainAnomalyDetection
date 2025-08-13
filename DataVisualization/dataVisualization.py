import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

class BridgeDataVisualizer:
    def __init__(self, data_file):
        """
        Initialize the data visualizer for bridge data (works with any dataset)
        """
        self.data_file = data_file
        self.df = None
        
        # Extract dataset name from filename for dynamic naming
        import os
        filename = os.path.basename(data_file)
        if 'cctp' in filename.lower():
            self.dataset_name = 'CCTP'
        elif 'across' in filename.lower():
            self.dataset_name = 'Across'
        else:
            # Extract name from filename (remove _bridge_data_labeled.csv)
            self.dataset_name = filename.replace('_bridge_data_labeled.csv', '').replace('_', ' ').title()
        
    def load_data(self):
        """
        Load the bridge dataset
        """
        print(f"Loading {self.dataset_name} bridge dataset...")
        self.df = pd.read_csv(self.data_file)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Check for tornado cash label column
        if 'tornado_cash_label' not in self.df.columns:
            print("Error: 'tornado_cash_label' column not found!")
            return None
        
        # Detect the correct amount column
        self.amount_column = None
        possible_amount_columns = ['amount_usd', 'input_amount_usd', 'output_amount_usd']
        for col in possible_amount_columns:
            if col in self.df.columns:
                self.amount_column = col
                break
        
        if self.amount_column is None:
            print("Error: No amount USD column found!")
            print("Available columns:", list(self.df.columns))
            return None
        
        print(f"Using amount column: {self.amount_column}")
        
        # Convert amount column to numeric (handle string values)
        print("Converting amount column to numeric...")
        self.df[self.amount_column] = pd.to_numeric(self.df[self.amount_column], errors='coerce')
        
        # Remove rows with NaN amounts (if any conversion failed)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.amount_column])
        if len(self.df) < initial_count:
            print(f"Removed {initial_count - len(self.df)} rows with invalid amount values")
            
        # Display basic statistics
        tornado_count = self.df['tornado_cash_label'].sum()
        legitimate_count = len(self.df) - tornado_count
        print(f"Tornado Cash transactions: {tornado_count} ({tornado_count/len(self.df)*100:.2f}%)")
        print(f"Legitimate transactions: {legitimate_count} ({legitimate_count/len(self.df)*100:.2f}%)")
        
        # Debug: Check amount distribution
        print(f"\nAmount Statistics:")
        print(f"Min amount: ${self.df[self.amount_column].min():.10f}")
        print(f"Max amount: ${self.df[self.amount_column].max():.2f}")
        print(f"Mean amount: ${self.df[self.amount_column].mean():.2f}")
        print(f"Median amount: ${self.df[self.amount_column].median():.2f}")
        
        # Show how many transactions are very small
        very_small = (self.df[self.amount_column] < 0.01).sum()
        small = (self.df[self.amount_column] < 1.0).sum()
        print(f"Transactions < $0.01: {very_small} ({very_small/len(self.df)*100:.1f}%)")
        print(f"Transactions < $1.00: {small} ({small/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def create_network_vs_amount_scatter(self, figsize=(12, 8), save_plot=True):
        """
        Create only the network vs amount scatter plot (top-left plot)
        """
        if self.df is None:
            self.load_data()
            
        print("\nCreating network vs amount distribution scatter plot...")
        
        # Separate legitimate and tornado cash transactions
        legitimate = self.df[self.df['tornado_cash_label'] == 0]
        tornado_cash = self.df[self.df['tornado_cash_label'] == 1]
        
        # Create single figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f'{self.dataset_name} Bridge Data: Transaction Distribution - Network vs Amount', 
                     fontsize=14, fontweight='bold')
       
        # Get unique networks for y-axis positioning
        networks = self.df['src_blockchain'].unique()
        network_positions = {network: i for i, network in enumerate(networks)}
        
        # Add small random jitter for better visualization
        legitimate_y = [network_positions[net] + np.random.normal(0, 0.1) for net in legitimate['src_blockchain']]
        tornado_y = [network_positions[net] + np.random.normal(0, 0.1) for net in tornado_cash['src_blockchain']]
        
        # Plot legitimate transactions in blue
        ax.scatter(legitimate[self.amount_column], legitimate_y, 
                   alpha=0.6, c='blue', s=20, label=f'Legitimate ({len(legitimate):,})')
        
        # Plot tornado cash transactions in red
        ax.scatter(tornado_cash[self.amount_column], tornado_y, 
                   alpha=0.8, c='red', s=30, label=f'Tornado Cash ({len(tornado_cash):,})')
        
        ax.set_xlabel('Amount (USD)', fontweight='bold')
        ax.set_ylabel('Source Network', fontweight='bold')
        ax.set_title('Transaction Distribution: Network vs Amount')
        ax.set_yticks(range(len(networks)))
        ax.set_yticklabels(networks)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for better amount visualization
        
        plt.tight_layout()
        
        if save_plot:
            import os
            filename = f'{self.dataset_name.lower()}_network_vs_amount_scatter.png'
            save_path = os.path.join(os.getcwd(), filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as '{save_path}'")
        
        plt.show()
    
    def create_network_vs_time_scatter(self, figsize=(14, 8), save_plot=True):
        """
        Create network vs time scatter plot showing transaction distribution over time
        """
        if self.df is None:
            self.load_data()
            
        print("\nCreating network vs time distribution scatter plot...")
        
        # Check if timestamp column exists
        if 'src_timestamp' not in self.df.columns:
            print("Error: 'src_timestamp' column not found!")
            print("Available columns:", list(self.df.columns))
            return
        
        # Convert timestamp to datetime
        self.df['datetime'] = pd.to_datetime(self.df['src_timestamp'], unit='s')
        
        # Separate legitimate and tornado cash transactions
        legitimate = self.df[self.df['tornado_cash_label'] == 0]
        tornado_cash = self.df[self.df['tornado_cash_label'] == 1]
        
        # Create single figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f'{self.dataset_name} Bridge Data: Transaction Distribution - Network vs Time', 
                     fontsize=14, fontweight='bold')
               
        # Get unique networks for y-axis positioning
        networks = self.df['src_blockchain'].unique()
        network_positions = {network: i for i, network in enumerate(networks)}
        
        # Add small random jitter for better visualization
        legitimate_y = [network_positions[net] + np.random.normal(0, 0.1) for net in legitimate['src_blockchain']]
        tornado_y = [network_positions[net] + np.random.normal(0, 0.1) for net in tornado_cash['src_blockchain']]
        
        # Plot legitimate transactions in blue
        ax.scatter(legitimate['datetime'], legitimate_y, 
                   alpha=0.6, c='blue', s=20, label=f'Legitimate ({len(legitimate):,})')
        
        # Plot tornado cash transactions in red
        ax.scatter(tornado_cash['datetime'], tornado_y, 
                   alpha=0.8, c='red', s=30, label=f'Tornado Cash ({len(tornado_cash):,})')
        
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Source Network', fontweight='bold')
        ax.set_title('Transaction Distribution: Network vs Time')
        ax.set_yticks(range(len(networks)))
        ax.set_yticklabels(networks)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for better time display
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add time range info
        time_range = f"From {legitimate['datetime'].min().strftime('%Y-%m-%d')} to {legitimate['datetime'].max().strftime('%Y-%m-%d')}"
        ax.text(0.02, 0.98, time_range, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            import os
            filename = f'{self.dataset_name.lower()}_network_vs_time_scatter.png'
            save_path = os.path.join(os.getcwd(), filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as '{save_path}'")
        
        plt.show()
        
    def create_detailed_network_analysis(self, figsize=(20, 12), save_plot=True):
        """
        Create detailed network analysis with multiple visualizations
        """
        if self.df is None:
            self.load_data()
            
        print("\nCreating detailed network analysis...")
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(f'Detailed {self.dataset_name} Network Analysis: Tornado Cash vs Legitimate Transactions', 
                     fontsize=18, fontweight='bold')
        
        
        # 1. Source Network Distribution
        ax1 = axes[0, 0]
        network_tornado = self.df.groupby(['src_blockchain', 'tornado_cash_label']).size().unstack(fill_value=0)
        network_tornado.plot(kind='bar', ax=ax1, color=['blue', 'red'], alpha=0.7)
        ax1.set_title('Source Network Distribution')
        ax1.set_ylabel('Transaction Count')
        ax1.legend(['Legitimate', 'Tornado Cash'])
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Destination Network Distribution
        ax2 = axes[0, 1]
        dst_network_tornado = self.df.groupby(['dst_blockchain', 'tornado_cash_label']).size().unstack(fill_value=0)
        dst_network_tornado.plot(kind='bar', ax=ax2, color=['blue', 'red'], alpha=0.7)
        ax2.set_title('Destination Network Distribution')
        ax2.set_ylabel('Transaction Count')
        ax2.legend(['Legitimate', 'Tornado Cash'])
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Cross-chain flow analysis
        ax3 = axes[0, 2]
        cross_chain = self.df[self.df['src_blockchain'] != self.df['dst_blockchain']]
        same_chain = self.df[self.df['src_blockchain'] == self.df['dst_blockchain']]
        
        flow_data = pd.DataFrame({
            'Flow Type': ['Cross-Chain', 'Same-Chain'],
            'Legitimate': [
                len(cross_chain[cross_chain['tornado_cash_label'] == 0]),
                len(same_chain[same_chain['tornado_cash_label'] == 0])
            ],
            'Tornado Cash': [
                len(cross_chain[cross_chain['tornado_cash_label'] == 1]),
                len(same_chain[same_chain['tornado_cash_label'] == 1])
            ]
        })
        
        flow_data.set_index('Flow Type').plot(kind='bar', ax=ax3, color=['blue', 'red'], alpha=0.7)
        ax3.set_title('Cross-Chain vs Same-Chain Flows')
        ax3.set_ylabel('Transaction Count')
        ax3.tick_params(axis='x', rotation=0)
        
        # 4. Amount distribution by network (violin plot)
        ax4 = axes[1, 0]
        
        # Prepare data for violin plot
        network_amount_data = []
        networks = self.df['src_blockchain'].unique()[:5]  # Top 5 networks
        
        for network in networks:
            net_data = self.df[self.df['src_blockchain'] == network]
            legitimate_amounts = net_data[net_data['tornado_cash_label'] == 0][self.amount_column]
            tornado_amounts = net_data[net_data['tornado_cash_label'] == 1][self.amount_column]
            
            # Add data points with labels
            for amount in legitimate_amounts:
                network_amount_data.append({'Network': network, 'Amount': amount, 'Type': 'Legitimate'})
            for amount in tornado_amounts:
                network_amount_data.append({'Network': network, 'Amount': amount, 'Type': 'Tornado Cash'})
        
        if network_amount_data:
            amount_df = pd.DataFrame(network_amount_data)
            sns.violinplot(data=amount_df, x='Network', y='Amount', hue='Type', 
                          ax=ax4, palette=['blue', 'red'])
            ax4.set_yscale('log')
            ax4.set_title('Amount Distribution by Network')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Time-based analysis (if timestamp available)
        ax5 = axes[1, 1]
        if 'src_timestamp' in self.df.columns:
            # Convert timestamp to datetime
            self.df['datetime'] = pd.to_datetime(self.df['src_timestamp'], unit='s')
            self.df['hour'] = self.df['datetime'].dt.hour
            
            hourly_tornado = self.df.groupby(['hour', 'tornado_cash_label']).size().unstack(fill_value=0)
            hourly_tornado.plot(ax=ax5, color=['blue', 'red'], alpha=0.7)
            ax5.set_title('Hourly Transaction Distribution')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('Transaction Count')
            ax5.legend(['Legitimate', 'Tornado Cash'])
        else:
            ax5.text(0.5, 0.5, 'Timestamp data not available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Temporal Analysis (No Data)')
        
        # 6. Address analysis
        ax6 = axes[1, 2]
        
        # Top addresses by transaction count
        top_depositors = self.df['depositor'].value_counts().head(10)
        tornado_by_depositor = self.df.groupby('depositor')['tornado_cash_label'].sum()
        
        # Create data for top depositors
        top_analysis = []
        for depositor in top_depositors.index:
            total_txs = top_depositors[depositor]
            tornado_txs = tornado_by_depositor.get(depositor, 0)
            legitimate_txs = total_txs - tornado_txs
            
            top_analysis.append({
                'Address': depositor[:10] + '...',  # Truncate for display
                'Legitimate': legitimate_txs,
                'Tornado Cash': tornado_txs
            })
        
        if top_analysis:
            top_df = pd.DataFrame(top_analysis)
            top_df.set_index('Address').plot(kind='barh', ax=ax6, color=['blue', 'red'], alpha=0.7)
            ax6.set_title('Top 10 Most Active Addresses')
            ax6.set_xlabel('Transaction Count')
        
        # 7. Amount percentile analysis
        ax7 = axes[2, 0]
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_data = []
        
        for p in percentiles:
            threshold = np.percentile(self.df[self.amount_column], p)
            above_threshold = self.df[self.df[self.amount_column] >= threshold]
            
            legitimate_count = len(above_threshold[above_threshold['tornado_cash_label'] == 0])
            tornado_count = len(above_threshold[above_threshold['tornado_cash_label'] == 1])
            
            percentile_data.append({
                'Percentile': f'{p}th',
                'Legitimate': legitimate_count,
                'Tornado Cash': tornado_count
            })
        
        perc_df = pd.DataFrame(percentile_data)
        perc_df.set_index('Percentile').plot(kind='bar', ax=ax7, color=['blue', 'red'], alpha=0.7)
        ax7.set_title('Transactions Above Amount Percentiles')
        ax7.set_ylabel('Transaction Count')
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Network pair analysis
        ax8 = axes[2, 1]
        
        # Most common network pairs
        self.df['network_pair'] = self.df['src_blockchain'] + ' → ' + self.df['dst_blockchain']
        top_pairs = self.df['network_pair'].value_counts().head(8)
        
        pair_analysis = []
        for pair in top_pairs.index:
            pair_data = self.df[self.df['network_pair'] == pair]
            legitimate_count = len(pair_data[pair_data['tornado_cash_label'] == 0])
            tornado_count = len(pair_data[pair_data['tornado_cash_label'] == 1])
            
            pair_analysis.append({
                'Pair': pair,
                'Legitimate': legitimate_count,
                'Tornado Cash': tornado_count
            })
        
        if pair_analysis:
            pair_df = pd.DataFrame(pair_analysis)
            pair_df.set_index('Pair').plot(kind='barh', ax=ax8, color=['blue', 'red'], alpha=0.7)
            ax8.set_title('Top Network Pairs')
            ax8.set_xlabel('Transaction Count')
        
        # 9. Summary statistics
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Calculate summary statistics
        total_txs = len(self.df)
        tornado_txs = self.df['tornado_cash_label'].sum()
        legitimate_txs = total_txs - tornado_txs
        
        avg_amount_legit = self.df[self.df['tornado_cash_label'] == 0][self.amount_column].mean()
        avg_amount_tornado = self.df[self.df['tornado_cash_label'] == 1][self.amount_column].mean()
        
        unique_networks = self.df['src_blockchain'].nunique()
        unique_addresses = self.df['depositor'].nunique()
        
        summary_text = f"""
        SUMMARY STATISTICS
        
        Total Transactions: {total_txs:,}
        Legitimate: {legitimate_txs:,} ({legitimate_txs/total_txs*100:.1f}%)
        Tornado Cash: {tornado_txs:,} ({tornado_txs/total_txs*100:.1f}%)
        
        Average Amount:
        Legitimate: ${avg_amount_legit:,.2f}
        Tornado Cash: ${avg_amount_tornado:,.2f}
        
        Networks: {unique_networks}
        Unique Addresses: {unique_addresses:,}
        
        Risk Ratio: 1 in {total_txs//tornado_txs if tornado_txs > 0 else 'N/A'}
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        if save_plot:
            import os
            filename = f'{self.dataset_name.lower()}_detailed_network_analysis.png'
            save_path = os.path.join(os.getcwd(), filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed analysis plot saved as '{save_path}'")
        
        plt.show()
    
    def create_risk_heatmap(self, figsize=(12, 8), save_plot=True):
        """
        Create a heatmap showing risk distribution across networks and amount ranges
        """
        if self.df is None:
            self.load_data()
            
        print("\nCreating risk heatmap...")
        
        # Define amount ranges
        amount_ranges = [
            (0, 100, '$0-$100'),
            (100, 1000, '$100-$1K'),
            (1000, 10000, '$1K-$10K'),
            (10000, 100000, '$10K-$100K'),
            (100000, float('inf'), '$100K+')
        ]
        
        # Create risk matrix
        networks = self.df['src_blockchain'].unique()
        risk_matrix = []
        
        for network in networks:
            network_data = self.df[self.df['src_blockchain'] == network]
            network_risk = []
            
            for min_amt, max_amt, _ in amount_ranges:
                range_mask = (network_data[self.amount_column] >= min_amt) & (network_data[self.amount_column] < max_amt)
                range_data = network_data[range_mask]
                
                if len(range_data) > 0:
                    risk_ratio = range_data['tornado_cash_label'].sum() / len(range_data) * 100
                else:
                    risk_ratio = 0
                    
                network_risk.append(risk_ratio)
            
            risk_matrix.append(network_risk)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        risk_df = pd.DataFrame(risk_matrix, 
                              index=networks, 
                              columns=[r[2] for r in amount_ranges])
        
        sns.heatmap(risk_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Tornado Cash Risk %'})
        
        ax.set_title('Tornado Cash Risk Heatmap: Network vs Amount Range', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Amount Range', fontweight='bold')
        ax.set_ylabel('Source Network', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            import os
            filename = f'{self.dataset_name.lower()}_risk_heatmap.png'
            save_path = os.path.join(os.getcwd(), filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk heatmap saved as '{save_path}'")
        
        plt.show()
    
    def generate_all_visualizations(self):
        """
        Generate both network vs amount and network vs time scatter plots
        """
        print("Starting network visualization...")
        
        # Load data
        self.load_data()
        
        # Create both scatter plots
        print("\n" + "="*50)
        print("GENERATING VISUALIZATION 1: NETWORK vs AMOUNT")
        print("="*50)
        self.create_network_vs_amount_scatter()
        
        print("\n" + "="*50)
        print("GENERATING VISUALIZATION 2: NETWORK vs TIME")
        print("="*50)
        self.create_network_vs_time_scatter()
        
        print("\nAll visualizations completed!")
        print("Generated files in current directory:")
        import os
        current_dir = os.getcwd()
        dataset_name = self.dataset_name.lower()
        print(f"1. {os.path.join(current_dir, f'{dataset_name}_network_vs_amount_scatter.png')} - Network vs Amount scatter plot")
        print(f"2. {os.path.join(current_dir, f'{dataset_name}_network_vs_time_scatter.png')} - Network vs Time scatter plot")


def main():
    """
    Main function to run data visualization
    """
    # Initialize visualizer - updated path since script is now in DataVisualization folder
    data_file = '../cctp_bridge_data_labeled.csv'  # Go up one level to find the CSV file
    data_file = '../across_bridge_data_labeled.csv'  # Go up one level to find the CSV file
  #  data_file = '../ccip_bridge_data_labeled.csv'  # Go up one level to find the CSV file
    
    visualizer = BridgeDataVisualizer(data_file)  # Updated class name
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()