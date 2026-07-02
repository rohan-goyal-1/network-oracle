"""Network construction helpers for simulated inquiry communities."""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from network_oracle.model import Params


def build_network(p: "Params", rng: np.random.Generator) -> nx.Graph:
    """Construct a connected graph with nodes labelled ``0..n_agents-1``."""
    n = p.n_agents
    topology = p.topology
    seed = int(rng.integers(1_000_000_000))

    if topology == "complete":
        graph = nx.complete_graph(n)
    elif topology == "cycle":
        graph = nx.cycle_graph(n)
    elif topology == "wheel":
        graph = nx.wheel_graph(n)
    elif topology == "star":
        graph = nx.star_graph(n - 1)
    elif topology == "er":
        prob = 0.3 if p.topo_param is None else float(p.topo_param)
        graph = nx.erdos_renyi_graph(n, prob, seed=seed)
        for _ in range(300):
            if nx.is_connected(graph):
                break
            seed += 1
            graph = nx.erdos_renyi_graph(n, prob, seed=seed)
    elif topology == "ws":
        degree = 4 if p.topo_param is None else int(p.topo_param)
        graph = nx.connected_watts_strogatz_graph(n, k=degree, p=0.1, seed=seed)
    elif topology == "ba":
        attachment = 2 if p.topo_param is None else int(p.topo_param)
        graph = nx.barabasi_albert_graph(n, m=attachment, seed=seed)
    else:
        raise ValueError(f"unknown topology {topology!r}")

    assert graph.number_of_nodes() == n, (topology, graph.number_of_nodes(), n)
    return graph


def neighborhood_matrix(graph: nx.Graph, n_agents: int) -> np.ndarray:
    """Return the neighbour-plus-self data-sharing matrix for a graph."""
    return nx.to_numpy_array(graph, nodelist=range(n_agents)) + np.eye(n_agents)
