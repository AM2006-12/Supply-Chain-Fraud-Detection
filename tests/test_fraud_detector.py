from src.graph_builder import GraphBuilder
from src.fraud_detector import FraudDetector
import networkx as nx


def make_simple_graph():
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=100)
    G.add_edge('B', 'C', weight=150)
    G.add_edge('C', 'A', weight=200)
    gb = GraphBuilder()
    gb.G = G
    return gb


def test_detect_fraud_statistical_only():
    gb = make_simple_graph()
    fd = FraudDetector(gb)
    results = fd.detect_fraud(use_gnn=False, use_statistical=True, use_graph_patterns=False)
    assert 'final_scores' in results
    assert 'node_scores' in results
    assert isinstance(results['final_scores'], dict)
    assert len(results['final_scores']) == gb.G.number_of_nodes()
