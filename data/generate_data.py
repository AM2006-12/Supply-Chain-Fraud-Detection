"""
Generate synthetic transaction data with fraud patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

class TransactionGenerator:
    def __init__(self, num_companies=50, num_transactions=1000):
        self.num_companies = num_companies
        self.num_transactions = num_transactions
        self.companies = self._generate_companies()
        
    def _generate_companies(self):
        """Generate company names"""
        prefixes = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega', 'Tech', 'Global', 
                   'Prime', 'Apex', 'Neo', 'Quantum', 'Stellar', 'Vertex', 'Zenith']
        suffixes = ['Corp', 'Industries', 'Systems', 'Solutions', 'Enterprises', 
                   'Ltd', 'Inc', 'Group', 'Holdings', 'Technologies']
        
        companies = []
        for i in range(self.num_companies):
            if i < len(prefixes):
                name = f"{prefixes[i]} {random.choice(suffixes)}"
            else:
                name = f"{random.choice(prefixes)} {random.choice(suffixes)} {i}"
            companies.append(name)
        return companies
    
    def generate_normal_transactions(self, count):
        """Generate legitimate transactions"""
        transactions = []
        start_date = datetime(2024, 1, 1)
        
        for _ in range(count):
            from_company = random.choice(self.companies)
            to_company = random.choice([c for c in self.companies if c != from_company])
            
            # Normal transaction amounts (log-normal distribution)
            amount = int(np.random.lognormal(13, 1))  # Mean ~500K
            
            # Random date in 2024
            days_offset = random.randint(0, 300)
            date = start_date + timedelta(days=days_offset)
            
            transactions.append({
                'transaction_id': f'TXN_{len(transactions)+1:05d}',
                'from_company': from_company,
                'to_company': to_company,
                'amount': amount,
                'date': date.strftime('%Y-%m-%d'),
                'category': random.choice(['Materials', 'Services', 'Equipment', 'Consulting']),
                'is_fraud': 0
            })
        
        return transactions
    
    def inject_circular_fraud(self, transactions, num_cycles=5):
        """Inject circular trading patterns"""
        for cycle_id in range(num_cycles):
            cycle_length = random.randint(3, 5)
            cycle_companies = random.sample(self.companies, cycle_length)
            
            base_amount = random.randint(800000, 2000000)
            start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 250))
            
            for i in range(cycle_length):
                from_comp = cycle_companies[i]
                to_comp = cycle_companies[(i + 1) % cycle_length]
                
                # Amount decreases slightly in cycle (profit extraction)
                amount = int(base_amount * (0.95 ** i))
                date = start_date + timedelta(days=i)
                
                transactions.append({
                    'transaction_id': f'FRAUD_CYCLE_{cycle_id}_{i}',
                    'from_company': from_comp,
                    'to_company': to_comp,
                    'amount': amount,
                    'date': date.strftime('%Y-%m-%d'),
                    'category': 'Materials',
                    'is_fraud': 1
                })
        
        return transactions
    
    def inject_shell_company_fraud(self, transactions, num_shells=3):
        """Inject shell company patterns"""
        for shell_id in range(num_shells):
            # Create shell company chain
            real_company = random.choice(self.companies)
            shell_1 = f"Shell_{shell_id}_A"
            shell_2 = f"Shell_{shell_id}_B"
            offshore = f"Offshore_{shell_id}"
            
            base_amount = random.randint(3000000, 8000000)
            start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 250))
            
            # Chain: Real -> Shell1 -> Shell2 -> Offshore
            chain = [
                (real_company, shell_1, base_amount),
                (shell_1, shell_2, int(base_amount * 0.9)),
                (shell_2, offshore, int(base_amount * 0.8))
            ]
            
            for i, (from_c, to_c, amt) in enumerate(chain):
                date = start_date + timedelta(days=i)
                transactions.append({
                    'transaction_id': f'FRAUD_SHELL_{shell_id}_{i}',
                    'from_company': from_c,
                    'to_company': to_c,
                    'amount': amt,
                    'date': date.strftime('%Y-%m-%d'),
                    'category': 'Services',
                    'is_fraud': 1
                })
        
        return transactions
    
    def inject_overpricing_fraud(self, transactions, count=10):
        """Inject overpriced transactions"""
        for _ in range(count):
            from_company = random.choice(self.companies)
            to_company = random.choice([c for c in self.companies if c != from_company])
            
            # Abnormally high amount
            amount = random.randint(10000000, 20000000)
            date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300))
            
            transactions.append({
                'transaction_id': f'FRAUD_OVERPRICE_{len(transactions)}',
                'from_company': from_company,
                'to_company': to_company,
                'amount': amount,
                'date': date.strftime('%Y-%m-%d'),
                'category': 'Materials',
                'is_fraud': 1
            })
        
        return transactions
    
    def generate_dataset(self):
        """Generate complete dataset with fraud"""
        print("Generating normal transactions...")
        transactions = self.generate_normal_transactions(
            self.num_transactions - 50  # Reserve space for fraud
        )
        
        print("Injecting circular fraud patterns...")
        transactions = self.inject_circular_fraud(transactions, num_cycles=5)
        
        print("Injecting shell company patterns...")
        transactions = self.inject_shell_company_fraud(transactions, num_shells=3)
        
        print("Injecting overpricing fraud...")
        transactions = self.inject_overpricing_fraud(transactions, count=10)
        
        # Shuffle
        random.shuffle(transactions)
        
        df = pd.DataFrame(transactions)
        
        print(f"\nDataset Summary:")
        print(f"Total transactions: {len(df)}")
        print(f"Fraudulent: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
        print(f"Legitimate: {(df['is_fraud']==0).sum()}")
        
        return df

if __name__ == "__main__":
    generator = TransactionGenerator(num_companies=50, num_transactions=1000)
    df = generator.generate_dataset()
    
    # Save to CSV
    output_path = 'sample_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print("\nSample data:")
    print(df.head(10))
