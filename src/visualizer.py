"""
Visualization tools for fraud detection
"""
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns

class FraudVisualizer:
    """
    Visualization tools for fraud detection results
    """
    def __init__(self, graph_builder):
        self.G = graph_builder.G
        self.graph_builder = graph_builder
    
    def create_interactive_network(self, fraud_scores, max_nodes=100, 
                                   min_risk_score=0.0, height='750px'):
        """
        Create interactive network visualization using PyVis
        
        Args:
            fraud_scores: Dictionary of node fraud scores
            max_nodes: Maximum nodes to display
            min_risk_score: Only show nodes with score >= this
            height: Height of visualization
        """
        # Filter high-risk nodes and their neighbors
        high_risk = [
            node for node, score in fraud_scores.items() 
            if score >= min_risk_score
        ]
        
        # Add neighbors for context
        nodes_to_show = set(high_risk)
        for node in high_risk[:20]:  # Limit to prevent explosion
            nodes_to_show.update(self.graph_builder.get_node_neighbors(node, hops=1))
        
        nodes_to_show = list(nodes_to_show)[:max_nodes]
        
        # Create subgraph
        subgraph = self.G.subgraph(nodes_to_show)
        
        # Create PyVis network
        net = Network(height=height, width='100%', directed=True, 
                     bgcolor='#222222', font_color='white')
        
        # Add nodes
        for node in subgraph.nodes():
            score = fraud_scores.get(node, 0)
            
            # Color based on risk (green -> yellow -> red)
            if score < 0.3:
                color = '#4CAF50'  # Green
            elif score < 0.6:
                color = '#FFC107'  # Yellow
            else:
                color = '#F44336'  # Red
            
            # Size based on degree
            size = 10 + self.G.degree(node) * 2
            
            net.add_node(
                node,
                label=node[:20],  # Truncate long names
                color=color,
                size=size,
                title=f"{node}<br>Risk: {score:.2f}<br>Connections: {self.G.degree(node)}"
            )
        
        # Add edges
        for u, v, data in subgraph.edges(data=True):
            weight = data.get('weight', 1)
            count = data.get('count', 1)
            
            net.add_edge(
                u, v,
                value=weight/1000000,  # Scale for visualization
                title=f"Amount: ₹{weight:,.0f}<br>Transactions: {count}"
            )
        
        # Physics settings
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          }
        }
        """)
        
        return net
    
    def create_plotly_network(self, fraud_scores, max_nodes=50):
        """
        Create network visualization using Plotly
        """
        # Get high-risk nodes
        top_nodes = sorted(
            fraud_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_nodes]
        
        nodes_to_show = [n for n, _ in top_nodes]
        subgraph = self.G.subgraph(nodes_to_show)
        
        # Calculate layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for u, v, data in subgraph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            weight = data.get('weight', 1)
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5 + weight/1000000, color='#888'),
                hoverinfo='text',
                text=f"{u} → {v}<br>Amount: ₹{weight:,.0f}",
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            score = fraud_scores.get(node, 0)
            node_color.append(score)
            node_size.append(10 + self.G.degree(node) * 3)
            node_text.append(f"{node}<br>Risk: {score:.2f}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[n[:15] for n in subgraph.nodes()],
            textposition='top center',
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(
                    title="Fraud Risk",
                    thickness=15,
                    len=0.7
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Fraud Detection Network',
                showlegend=False,
                hovermode='closest',
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700
            )
        )
        
        return fig
    
    def plot_risk_distribution(self, fraud_scores):
        """Plot distribution of fraud risk scores"""
        scores = list(fraud_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=50,
            marker_color='rgba(255, 100, 100, 0.7)',
            name='Risk Distribution'
        ))
        
        fig.update_layout(
            title='Fraud Risk Score Distribution',
            xaxis_title='Risk Score',
            yaxis_title='Number of Entities',
            plot_bgcolor='#f8f9fa',
            height=400
        )
        
        return fig
    
    def plot_top_risks(self, fraud_scores, top_n=10):
        """Bar chart of top risky entities"""
        top_risks = sorted(
            fraud_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        nodes, scores = zip(*top_risks)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=[n[:30] for n in nodes],  # Truncate names
            x=scores,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='YlOrRd',
                showscale=False
            )
        ))
        
        fig.update_layout(
            title=f'Top {top_n} High-Risk Entities',
            xaxis_title='Risk Score',
            yaxis_title='Entity',
            height=400,
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def visualize_cycle(self, cycle_nodes):
        """Visualize a specific circular trading pattern"""
        # Create cycle subgraph
        cycle_graph = nx.DiGraph()
        
        for i in range(len(cycle_nodes)):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % len(cycle_nodes)]
            
            if self.G.has_edge(u, v):
                weight = self.G[u][v]['weight']
                cycle_graph.add_edge(u, v, weight=weight)
        
        # Circular layout
        pos = nx.circular_layout(cycle_graph)
        
        # Create plotly figure
        edge_x = []
        edge_y = []
        edge_text = []
        
        for u, v, data in cycle_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"₹{data['weight']:,.0f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=3, color='red'),
            mode='lines',
            showlegend=False
        )
        
        node_x = [pos[node][0] for node in cycle_graph.nodes()]
        node_y = [pos[node][1] for node in cycle_graph.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(cycle_graph.nodes()),
            textposition='top center',
            marker=dict(size=30, color='orange', line=dict(width=2, color='white'))
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='Circular Trading Pattern',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig
    
    def create_summary_dashboard(self, results):
        """Create summary statistics for dashboard"""
        summary = {
            'Total Entities': self.G.number_of_nodes(),
            'Total Transactions': self.G.number_of_edges(),
            'High Risk Entities': len(results['high_risk_entities']),
            'Suspicious Cycles': len(results['cycles']),
            'Fraud Rings Detected': sum(1 for c in results['communities'] if c['risk_score'] > 0.5)
        }
        
        return summary
