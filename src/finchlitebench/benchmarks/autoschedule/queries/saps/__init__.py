"""SAPS subgraph-matching query shapes, one query per shape family.

Each query counts homomorphisms of a small query graph into one random
undirected data graph G: one einsum index per query vertex and one ``G``
operand per query edge, summed to a scalar.  The numeric suffix is the number
of query edges, as in SAPS.
"""
