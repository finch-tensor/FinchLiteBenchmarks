"""Tests for the ported G-CARE parsers, using the toy graph from SAPS's
``SubgraphMatchingTestGenerator`` (label A on vertices 0 and 2, label B on
vertex 1, edges 0->1, 0->2, 2->1)."""

import numpy as np
import pytest

from finchtensorbench import gcare

GRAPH = """t # 0
v 0 10
v 1 11
v 2 10
e 0 1 0
e 0 2 0
e 2 1 0
"""

# name -> (query file, expected count from SAPS)
QUERIES = {
    "all_edges": ("t # 0\nv 0 -1 -1\nv 1 -1 -1\ne 0 1 0\n", 3),
    "label_a_count": ("t # 0\nv 0 10 -1\n", 2),
    "a_to_b": ("t # 0\nv 0 10 -1\nv 1 11 -1\ne 0 1 0\n", 2),
    "b_to_a": ("t # 0\nv 0 11 -1\nv 1 10 -1\ne 0 1 0\n", 0),
    "pinned": ("t # 0\nv 0 -1 0\nv 1 -1 -1\ne 0 1 0\n", 2),
    "missing_label": ("t # 0\nv 0 99 -1\nv 1 -1 -1\ne 0 1 0\n", 0),
    "missing_edge_label": ("t # 0\nv 0 -1 -1\nv 1 -1 -1\ne 0 1 7\n", 0),
}


@pytest.fixture
def toy(tmp_path):
    graph_path = tmp_path / "toy.txt"
    graph_path.write_text(GRAPH)
    graph = gcare.parse_graph(graph_path)
    queries = {}
    for name, (text, expected) in QUERIES.items():
        path = tmp_path / f"{name}.txt"
        path.write_text(text)
        queries[name] = gcare.parse_query(
            path, continuous_ids=graph.continuous_ids, name=name, ground_truth=expected
        )
    return graph, queries


def _dense(graph, name):
    ndim, coords = gcare.query_tensor_coo(graph, name)
    array = np.zeros((graph.num_vertices,) * ndim)
    array[coords] = 1.0
    return array


def test_parse_graph(toy):
    graph, _ = toy
    assert graph.num_vertices == 3 and graph.continuous_ids
    assert sorted(graph.vectors) == ["V10", "V11"]
    assert graph.vectors["V10"].tolist() == [0, 2]
    rows, cols = graph.matrices["E0"]
    assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(0, 1), (0, 2), (2, 1)]


def test_query_expressions(toy):
    _, queries = toy
    query = queries["a_to_b"]
    assert query.saps_expr() == "S[] += V10[v_0] * V11[v_1] * E0[v_0,v_1]"
    assert query.subscripts() == "a,b,ab->"
    assert queries["pinned"].saps_expr() == "S[] += P0[v_0] * E0[v_0,v_1]"


@pytest.mark.parametrize("name", list(QUERIES))
def test_counts_numpy(toy, name):
    graph, queries = toy
    query = queries[name]
    operands = [_dense(graph, op.tensor) for op in query.operands]
    assert np.einsum(query.subscripts(), *operands) == query.ground_truth


def test_noncontiguous_ids_add_existence_vector(tmp_path):
    path = tmp_path / "gaps.txt"
    path.write_text("t # 0\nv 0 1\nv 5 1\ne 0 5 0\n")
    graph = gcare.parse_graph(path)
    assert not graph.continuous_ids and graph.vectors["C"].tolist() == [0, 5]
    query_path = tmp_path / "q.txt"
    query_path.write_text("t # 0\nv 0 -1 -1\n")
    query = gcare.parse_query(query_path, continuous_ids=False)
    assert query.saps_expr() == "S[] += C[v_0]"


def test_sgm_graph_is_undirected_and_queries_label_every_vertex(tmp_path):
    # A triangle 0-1-2 plus the edge 2-3, given once per edge (and 0-1 twice).
    path = tmp_path / "g.graph"
    path.write_text(
        "t 4 4\nv 0 7 2\nv 1 7 2\nv 2 8 3\nv 3 7 1\n"
        "e 0 1 0\ne 1 2 0\ne 0 2 0\ne 2 3 0\ne 1 0 0\n"
    )
    graph = gcare.parse_sgm_graph(path, "toy")
    assert graph.continuous_ids and graph.vectors["V7"].tolist() == [0, 1, 3]
    assert len(graph.matrices["E0"][0]) == 8
    query_path = tmp_path / "q.graph"
    query_path.write_text("t 2 1\nv 0 7 1\nv 1 8 1\ne 0 1 0\n")
    query = gcare.parse_sgm_query(query_path)
    assert query.saps_expr() == "S[] += V7[v_0] * V8[v_1] * E0[v_0,v_1]"
    operands = [_dense(graph, op.tensor) for op in query.operands]
    # Each label-7 neighbour of vertex 2 (0, 1 and 3) matches once.
    assert np.einsum(query.subscripts(), *operands) == 3


@pytest.mark.slow
@pytest.mark.parametrize("config", ["default", "galley"])
def test_counts_finch(toy, config):
    finch = pytest.importorskip("finch")
    from finchtensorbench.benchmarks.autoschedule import build_scheduler, parse_config
    from finchtensorbench.benchmarks.subgraph.subgraph_matching import (
        GraphTensors,
        query_program,
    )

    graph, queries = toy
    tensors = GraphTensors(graph)
    for query in queries.values():
        ctx = build_scheduler(parse_config(config))
        result = finch.to_numpy(finch.compute(query_program(query, tensors), ctx=ctx))
        assert float(np.asarray(result).ravel()[0]) == query.ground_truth, query.name
