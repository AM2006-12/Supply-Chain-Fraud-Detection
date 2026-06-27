"""
Main fraud detection pipeline
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

class FraudDetector:
    """
    Main fraud detection system combining multiple approaches
    """
    def __init__(self, graph_builder, gnn_model=None):
        self.graph_builder = graph_builder
        self.gnn_model = gnn_model
        self.fraud_scores = {}
        
    def detect_fraud(self, use_gnn=True, use_statistical=True, use_graph_patterns=True, fast_mode=False):
        """
        Run comprehensive fraud detection
        
        Args:
            use_gnn: Use GNN model predictions
            use_statistical: Use statistical anomaly detection
            use_graph_patterns: Use graph pattern analysis
            fast_mode: If True, avoid expensive graph pattern searches and use faster heuristics
        
        Returns:
            Dictionary of fraud scores and detections
        """
        print("\n=== Running Fraud Detection ===")
        
        results = {
            'node_scores': {},
            'cycles': [],
            'communities': [],
            'anomalies': [],
            'high_risk_entities': []
        }
        
        # 1. Graph pattern analysis (may be expensive on large graphs)
        if use_graph_patterns:
            print("\n1. Analyzing graph patterns...")
            node_count = self.graph_builder.G.number_of_nodes()
            # Fast mode or very large graphs: use tight limits and shorter timeouts
            if fast_mode or node_count > 500:
                print(f"Warning: Running in fast mode or graph is large ({node_count} nodes); limiting pattern detection for performance.")
                cycles = self.graph_builder.detect_cycles(max_length=6, max_cycles=100, max_time=3.0)
                # For communities, if graph is huge, skip heavy community detection
                try:
                    if node_count > 1500 and fast_mode:
                        communities = []
                    else:
                        communities = self.graph_builder.detect_communities()
                except Exception as e:
                    print(f"Community detection error (skipping): {e}")
                    communities = []
            else:
                cycles = self.graph_builder.detect_cycles()
                communities = self.graph_builder.detect_communities()
            results['cycles'] = cycles
            results['communities'] = communities
            
            # Score nodes based on pattern participation
            pattern_scores = self._score_from_patterns(
                results['cycles'], 
                results['communities']
            )
            results['node_scores']['pattern'] = pattern_scores
        
        # 2. Statistical anomaly detection
        if use_statistical:
            print("\n2. Running statistical anomaly detection...")
            node_features = self.graph_builder.compute_node_features()
            anomaly_scores = self._detect_statistical_anomalies(node_features)
            results['node_scores']['statistical'] = anomaly_scores
            results['anomalies'] = [
                node for node, score in anomaly_scores.items() if score > 0.7
            ]
        
        # 3. GNN predictions
        if use_gnn and self.gnn_model is not None:
            print("\n3. Running GNN model...")
            # This would be populated if model is trained
            # For now, we'll combine other scores
            pass
        
        # 4. Combine scores
        print("\n4. Computing final risk scores...")
        final_scores = self._combine_scores(results['node_scores'])
        results['final_scores'] = final_scores
        
        # 5. Identify high-risk entities
        threshold = 0.6
        results['high_risk_entities'] = [
            {'node': node, 'score': score, 'rank': rank+1}
            for rank, (node, score) in enumerate(
                sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            )
            if score > threshold
        ]
        
        print(f"\nDetection Summary:")
        print(f"  - Suspicious cycles found: {len(results['cycles'])}")
        print(f"  - Suspicious communities: {sum(1 for c in results['communities'] if c['risk_score'] > 0.5)}")
        print(f"  - Statistical anomalies: {len(results['anomalies'])}")
        print(f"  - High-risk entities: {len(results['high_risk_entities'])}")
        
        return results
    
    def _score_from_patterns(self, cycles, communities):
        """Score nodes based on graph pattern participation"""
        scores = {}
        
        # Initialize all nodes
        for node in self.graph_builder.G.nodes():
            scores[node] = 0.0
        
        # Score based on cycle participation
        for cycle in cycles:
            risk = cycle['risk_score']
            for node in cycle['nodes']:
                scores[node] = max(scores[node], risk * 0.8)
        
        # Score based on community suspiciousness
        for community in communities:
            risk = community['risk_score']
            for node in community['members']:
                current = scores.get(node, 0)
                scores[node] = max(current, risk * 0.6)
        
        return scores
    
    def _detect_statistical_anomalies(self, node_features):
        """Detect anomalies using Isolation Forest"""
        # Convert features to matrix
        nodes = list(node_features.keys())
        feature_keys = sorted(list(node_features[nodes[0]].keys()))
        
        X = np.array([
            [node_features[node][key] for key in feature_keys]
            for node in nodes
        ])
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        clf = IsolationForest(
            contamination=0.1,  # Expect ~10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        predictions = clf.fit_predict(X_scaled)
        anomaly_scores = clf.score_samples(X_scaled)
        
        # Normalize scores to [0, 1]
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        anomaly_scores = 1 - anomaly_scores  # Invert so high = anomalous
        
        # Create score dictionary
        scores = {}
        for i, node in enumerate(nodes):
            if predictions[i] == -1:  # Anomaly
                scores[node] = float(anomaly_scores[i])
            else:
                scores[node] = float(anomaly_scores[i] * 0.3)  # Lower score for normal
        
        return scores
    
    def _combine_scores(self, score_dict):
        """Combine multiple scoring methods"""
        all_nodes = set()
        for scores in score_dict.values():
            all_nodes.update(scores.keys())
        
        final_scores = {}
        
        for node in all_nodes:
            scores = []
            weights = []
            
            if 'pattern' in score_dict:
                scores.append(score_dict['pattern'].get(node, 0))
                weights.append(0.4)
            
            if 'statistical' in score_dict:
                scores.append(score_dict['statistical'].get(node, 0))
                weights.append(0.3)
            
            if 'gnn' in score_dict:
                scores.append(score_dict['gnn'].get(node, 0))
                weights.append(0.3)
            
            # Weighted average
            if scores:
                final_scores[node] = sum(s*w for s, w in zip(scores, weights)) / sum(weights)
            else:
                final_scores[node] = 0.0
        
        return final_scores
    
    def generate_report(self, results):
        """Generate fraud detection report"""
        report = {
            'summary': {
                'total_entities': self.graph_builder.G.number_of_nodes(),
                'total_transactions': self.graph_builder.G.number_of_edges(),
                'high_risk_count': len(results['high_risk_entities']),
                'suspicious_cycles': len(results['cycles']),
                'fraud_rings': sum(1 for c in results['communities'] if c['risk_score'] > 0.5)
            },
            'top_risks': results['high_risk_entities'][:10],
            'top_cycles': results['cycles'][:5],
            'suspicious_communities': [
                c for c in results['communities'] if c['risk_score'] > 0.5
            ][:3]
        }
        
        return report
    
    def get_entity_details(self, entity_name, results):
        """Get detailed information about a specific entity"""
        G = self.graph_builder.G
        
        if entity_name not in G.nodes():
            return None
        
        # Get neighbors
        predecessors = list(G.predecessors(entity_name))
        successors = list(G.successors(entity_name))
        
        # Get transactions
        incoming_tx = [
            {
                'from': u,
                'amount': G[u][entity_name]['weight'],
                'count': G[u][entity_name]['count']
            }
            for u in predecessors
        ]
        
        outgoing_tx = [
            {
                'to': v,
                'amount': G[entity_name][v]['weight'],
                'count': G[entity_name][v]['count']
            }
            for v in successors
        ]
        
        # Get risk score
        risk_score = results['final_scores'].get(entity_name, 0)
        
        # Check pattern participation
        in_cycles = [c for c in results['cycles'] if entity_name in c['nodes']]
        in_communities = [c for c in results['communities'] if entity_name in c['members']]
        
        details = {
            'entity': entity_name,
            'risk_score': risk_score,
            'total_received': sum(tx['amount'] for tx in incoming_tx),
            'total_sent': sum(tx['amount'] for tx in outgoing_tx),
            'num_suppliers': len(predecessors),
            'num_customers': len(successors),
            'incoming_transactions': incoming_tx,
            'outgoing_transactions': outgoing_tx,
            'participates_in_cycles': len(in_cycles),
            'cycle_details': in_cycles[:3],
            'community_membership': in_communities[0] if in_communities else None
        }
        
        return details
