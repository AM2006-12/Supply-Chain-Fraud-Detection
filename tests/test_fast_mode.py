from src.graph_builder import GraphBuilder
from src.fraud_detector import FraudDetector
import networkx as nx


def test_detect_fraud_fast_mode_large_graph():
    # Build a larger graph (but not too huge) to test fast_mode
    G = nx.DiGraph()
    for i in range(800):
        G.add_node(f'N{i}')
    for i in range(1600):
        u = f'N{(i*3) % 800}'
        v = f'N{(i*7+5) % 800}'
        G.add_edge(u, v, weight=(i+1)*10)

    gb = GraphBuilder()
    gb.G = G
    fd = FraudDetector(gb)

    # Run in fast_mode to ensure cycle detection times out early and returns
    results = fd.detect_fraud(use_gnn=False, use_statistical=True, use_graph_patterns=True, fast_mode=True)

    assert 'final_scores' in results
    assert isinstance(results['final_scores'], dict)
    # Should not hang; check we have node scores for at least some nodes
    assert len(results['final_scores']) > 0
