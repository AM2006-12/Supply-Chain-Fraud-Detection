import networkx as nx
from src.graph_builder import GraphBuilder


def test_detect_cycles_limited():
    # Create small graph with a known cycle
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=100)
    G.add_edge('B', 'C', weight=200)
    G.add_edge('C', 'A', weight=300)

    gb = GraphBuilder()
    gb.G = G

    cycles = gb.detect_cycles(max_length=5, max_cycles=10)
    assert isinstance(cycles, list)
    assert len(cycles) >= 1
    assert any(set(['A','B','C']).issubset(set(c['nodes'])) for c in cycles)


def test_compute_node_features_large_graph():
    # Create a larger graph to trigger approximate betweenness
    G = nx.DiGraph()
    for i in range(300):
        G.add_node(f'node_{i}')
    # add some edges
    for i in range(500):
        u = f'node_{i % 300}'
        v = f'node_{(i+1) % 300}'
        G.add_edge(u, v, weight=(i+1)*100)

    gb = GraphBuilder()
    gb.G = G

    features = gb.compute_node_features()
    # Basic assertions
    assert isinstance(features, dict)
    assert 'node_0' in features
    assert 'betweenness' in features['node_0']
