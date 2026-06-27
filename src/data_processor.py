"""
Data preprocessing and cleaning
"""
import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.df = None
        
    def load_data(self, file_path):
        """Load transaction data from CSV"""
        self.df = pd.read_csv(file_path)
        print(f"Loaded {len(self.df)} transactions")
        return self.df
    
    def clean_data(self, df=None):
        """Clean and validate data"""
        if df is None:
            df = self.df
            
        print("Cleaning data...")
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['transaction_id'])
        print(f"Removed {original_len - len(df)} duplicates")
        
        # Handle missing values
        df = df.dropna(subset=['from_company', 'to_company', 'amount'])
        
        # Normalize company names
        df['from_company'] = df['from_company'].str.strip().str.lower()
        df['to_company'] = df['to_company'].str.strip().str.lower()
        
        # Ensure amount is numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        self.df = df
        print(f"Clean data: {len(df)} transactions")
        return df
    
    def compute_basic_features(self, df=None):
        """Compute transaction-level features"""
        if df is None:
            df = self.df
            
        print("Computing features...")
        
        # Transaction velocity (how many transactions per company)
        company_counts = df['from_company'].value_counts()
        df['from_transaction_count'] = df['from_company'].map(company_counts)
        
        company_counts = df['to_company'].value_counts()
        df['to_transaction_count'] = df['to_company'].map(company_counts)
        
        # Average amounts
        company_avg = df.groupby('from_company')['amount'].mean()
        df['from_avg_amount'] = df['from_company'].map(company_avg)
        
        company_avg = df.groupby('to_company')['amount'].mean()
        df['to_avg_amount'] = df['to_company'].map(company_avg)
        
        # Amount deviation from average
        df['amount_deviation'] = (df['amount'] - df['from_avg_amount']) / (df['from_avg_amount'] + 1)
        
        # Large transaction flag
        df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        self.df = df
        return df
    
    def get_company_list(self, df=None):
        """Get unique list of companies"""
        if df is None:
            df = self.df
            
        companies = set(df['from_company']) | set(df['to_company'])
        return sorted(list(companies))
    
    def get_summary_stats(self, df=None):
        """Get dataset summary statistics"""
        if df is None:
            df = self.df
            
        stats = {
            'total_transactions': len(df),
            'unique_companies': len(self.get_company_list(df)),
            'total_volume': df['amount'].sum(),
            'avg_amount': df['amount'].mean(),
            'median_amount': df['amount'].median(),
            'max_amount': df['amount'].max(),
            'min_amount': df['amount'].min(),
        }
        
        if 'is_fraud' in df.columns:
            stats['fraud_count'] = df['is_fraud'].sum()
            stats['fraud_percentage'] = df['is_fraud'].mean() * 100
        
        return stats
    
