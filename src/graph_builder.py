"""
Build graph structure from transaction data
"""
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
import community as community_louvain
import time

class GraphBuilder:
    def __init__(self):
        self.G = None
        self.node_to_idx = {}
        self.idx_to_node = {}
        # Cache for detected cycles to avoid repeated expensive searches
        self._cycles_cache = None
        
    def build_graph(self, df):
        """Build directed graph from transactions"""
        print("Building graph...")
        self.G = nx.DiGraph()
        
        # Add all unique companies as nodes
        companies = set(df['from_company']) | set(df['to_company'])
        for company in companies:
            self.G.add_node(company, node_type='company')
        
        # Add edges (transactions)
        for _, row in df.iterrows():
            from_c = row['from_company']
            to_c = row['to_company']
            amount = row['amount']
            
            if self.G.has_edge(from_c, to_c):
                # Multiple transactions between same parties
                self.G[from_c][to_c]['weight'] += amount
                self.G[from_c][to_c]['count'] += 1
                self.G[from_c][to_c]['amounts'].append(amount)
            else:
                self.G.add_edge(
                    from_c, to_c,
                    weight=amount,
                    count=1,
                    amounts=[amount]
                )
        
        # Create node index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(self.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        print(f"Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G
    
    def compute_node_features(self):
        """Compute features for each node (company)"""
        print("Computing node features...")
        features = {}
        
        # Pre-compute some graph metrics
        try:
            pagerank = nx.pagerank(self.G, max_iter=100)
        except Exception:
            pagerank = {node: 1.0/self.G.number_of_nodes() for node in self.G.nodes()}
        
        try:
            # For large graphs, compute approximate betweenness using sampling to save time
            if self.G.number_of_nodes() > 200:
                k = min(50, max(1, self.G.number_of_nodes() - 1))
                betweenness = nx.betweenness_centrality(self.G, k=k, seed=42)
            else:
                betweenness = nx.betweenness_centrality(self.G)
        except Exception:
            betweenness = {node: 0.0 for node in self.G.nodes()}
        
        # For clustering, convert to undirected
        G_undirected = self.G.to_undirected()
        try:
            clustering = nx.clustering(G_undirected)
        except Exception:
            clustering = {node: 0.0 for node in self.G.nodes()}
        
        for node in self.G.nodes():
            # Degree features
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            
            # Transaction volumes
            total_received = sum([self.G[u][node]['weight'] for u in self.G.predecessors(node)]) if in_deg > 0 else 0
            total_sent = sum([self.G[node][v]['weight'] for v in self.G.successors(node)]) if out_deg > 0 else 0
            
            # Average transaction sizes
            avg_received = total_received / in_deg if in_deg > 0 else 0
            avg_sent = total_sent / out_deg if out_deg > 0 else 0
            
            # Check if participates in cycles
            participates_in_cycle = self._check_cycle_participation(node)
            
            features[node] = {
                'in_degree': in_deg,
                'out_degree': out_deg,
                'total_degree': in_deg + out_deg,
                'total_received': total_received,
                'total_sent': total_sent,
                'avg_received': avg_received,
                'avg_sent': avg_sent,
                'net_flow': total_received - total_sent,
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'clustering': clustering.get(node, 0),
                'in_cycle': int(participates_in_cycle),
                'single_transaction_vendor': int(in_deg == 1 and out_deg == 0),
            }
        
        return features
    
    def _check_cycle_participation(self, node):
        """Check if node participates in any cycle (uses cached cycles when available)"""
        try:
            # Populate cache if needed (limited search to avoid heavy computation)
            if self._cycles_cache is None:
                # Use a reasonable default bound for caching
                self._cycles_cache = self.detect_cycles(max_length=10, max_cycles=500)
            for cycle in self._cycles_cache:
                if node in cycle['nodes']:
                    return True
            return False
        except Exception:
            return False
    
    def detect_cycles(self, max_length=10, max_cycles=500, max_time=5.0):
        """Detect circular trading patterns with limits to avoid long-running computation

        Args:
            max_length: maximum cycle length to consider
            max_cycles: maximum number of cycles to detect before truncating
            max_time: maximum time in seconds to spend searching cycles
        """
        print("Detecting cycles (limited search)...")
        cycles = []
        found = 0
        start_time = time.time()
        
        try:
            for cycle in nx.simple_cycles(self.G):
                # Time cutoff
                if time.time() - start_time > max_time:
                    print(f"Cycle detection time cutoff after {max_time} seconds")
                    break

                if 3 <= len(cycle) <= max_length:
                    # Calculate total amount in cycle
                    cycle_amount = sum([
                        self.G[cycle[i]][cycle[(i+1) % len(cycle)]]['weight']
                        for i in range(len(cycle))
                    ])
                    
                    cycles.append({
                        'nodes': cycle,
                        'length': len(cycle),
                        'total_amount': cycle_amount,
                        'risk_score': self._compute_cycle_risk(cycle)
                    })
                    found += 1
                    if found >= max_cycles:
                        print(f"Cycle detection truncated after {max_cycles} cycles for performance.")
                        break
        except Exception as e:
            print(f"Warning: Cycle detection limited due to: {e}")
        
        # Sort by risk score and cache
        cycles = sorted(cycles, key=lambda x: x['risk_score'], reverse=True)
        self._cycles_cache = cycles
        print(f"Found {len(cycles)} cycles (truncated={found >= max_cycles})")
        return cycles
    
    def _compute_cycle_risk(self, cycle):
        """Compute risk score for a cycle"""
        # Shorter cycles are more suspicious
        length_penalty = 1.0 / len(cycle)
        
        # Calculate amount flow
        amounts = [
            self.G[cycle[i]][cycle[(i+1) % len(cycle)]]['weight']
            for i in range(len(cycle))
        ]
        total_amount = sum(amounts)
        
        # Amount similarity (if amounts are similar, more suspicious)
        amount_std = np.std(amounts) if len(amounts) > 1 else 0
        amount_similarity = 1.0 / (1.0 + amount_std / (np.mean(amounts) + 1))
        
        # Combine factors
        risk_score = (length_penalty * 0.4 + 
                     (total_amount / 10000000) * 0.3 + 
                     amount_similarity * 0.3)
        
        return min(risk_score, 1.0)
    
    def detect_communities(self):
        """Detect tightly connected groups (fraud rings)"""
        print("Detecting communities...")
        
        # Convert to undirected for community detection
        G_undirected = self.G.to_undirected()
        
        try:
            communities = community_louvain.best_partition(G_undirected)
        except:
            print("Warning: Community detection failed, using default")
            communities = {node: 0 for node in self.G.nodes()}
        
        # Analyze each community
        community_analysis = []
        for comm_id in set(communities.values()):
            members = [n for n, c in communities.items() if c == comm_id]
            
            if len(members) < 2:
                continue
                
            subgraph = self.G.subgraph(members)
            
            internal_edges = subgraph.number_of_edges()
            total_edges = sum([self.G.degree(n) for n in members])
            external_edges = total_edges - 2 * internal_edges
            
            internal_ratio = internal_edges / (internal_edges + external_edges + 1)
            
            # Calculate total volume
            total_volume = sum([
                self.G[u][v]['weight'] 
                for u, v in subgraph.edges()
            ])
            
            community_analysis.append({
                'community_id': comm_id,
                'members': members,
                'size': len(members),
                'internal_ratio': internal_ratio,
                'total_volume': total_volume,
                'risk_score': internal_ratio if internal_ratio > 0.6 else 0
            })
        
        # Sort by risk
        community_analysis = sorted(
            community_analysis, 
            key=lambda x: x['risk_score'], 
            reverse=True
        )
        
        print(f"Found {len(community_analysis)} communities")
        return community_analysis
    
    def get_node_neighbors(self, node, hops=2):
        """Get k-hop neighbors of a node"""
        neighbors = set()
        current_level = {node}
        
        for _ in range(hops):
            next_level = set()
            for n in current_level:
                # Add successors and predecessors
                next_level.update(self.G.successors(n))
                next_level.update(self.G.predecessors(n))
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.discard(node)
        return list(neighbors)
