"""Symbolic elimination of the same box-reduced KKT graph in each ordering."""
import hashlib
import heapq
from itertools import combinations
import struct

ORDERINGS = ("generic_minimum_degree", "root_last_minimum_degree", "root_last", "root_first")

from model import indices


def graph(q):
    H, E = q.lower_hessian, q.jacobian
    adjacency = [set() for _ in range(q.layout.n + q.layout.m)]

    def connect(i, j):
        if i != j:
            adjacency[i].add(j)
            adjacency[j].add(i)

    for i in range(H.shape[0]):
        for j in H.indices[H.indptr[i]:H.indptr[i + 1]]:
            connect(i, int(j))
    for i in range(E.shape[0]):
        for j in E.indices[E.indptr[i]:E.indptr[i + 1]]:
            connect(q.layout.n + i, int(j))
    return adjacency


def eliminate(pattern, order=None, *, defer=()):
    """Static symmetric symbolic elimination, including a diagonal per column.

    With no permutation, use exact minimum degree, breaking ties by vertex ID.
    The deferred vertices, when supplied, come after the other vertices.
    Rows are adjacency sets only. Each recorded front includes its pivot.
    """
    adjacency = [set(row) for row in pattern]
    size = len(adjacency)
    entries = size + sum(map(len, adjacency)) // 2
    deferred = set(defer)

    def priority(i):
        return (i in deferred, len(adjacency[i]), i)

    if order is None:
        heap = [priority(i) for i in range(size)]
        heapq.heapify(heap)
    elif sorted(order) != list(range(size)):
        raise ValueError("not a permutation of the KKT vertices")

    permutation, remaining = [], set(range(size))
    factor_nnz, max_front = 0, 0
    for k in range(size):
        if order is None:
            while True:
                _, degree, pivot = heapq.heappop(heap)
                if pivot in remaining and degree == len(adjacency[pivot]):
                    break
        else:
            pivot = order[k]
        permutation.append(pivot)
        remaining.remove(pivot)
        neighbors = adjacency[pivot]
        front = len(neighbors) + 1
        factor_nnz += front
        max_front = max(max_front, front)
        for vertex in neighbors:
            adjacency[vertex].remove(pivot)
        for left, right in combinations(neighbors, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
        if order is None:
            for vertex in neighbors:
                heapq.heappush(heap, priority(vertex))
        neighbors.clear()
    return {
        "permutation": permutation, "factor_lower_nnz_including_diagonal": factor_nnz,
        "fill_edges": factor_nnz - entries, "largest_front_including_pivot": max_front,
    }


def branch_order(L):
    permutation = []
    for j in range(L.J):
        for k in range(L.N, 0, -1):
            permutation.extend(indices(L.stage(j, k)))
            permutation.extend(L.n + row for row in indices(L.eq(j, k - 1)))
    return permutation


def compare(q):
    pattern = graph(q)
    L = q.layout
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", L.n, L.m))
    for i, row in enumerate(pattern):
        for j in sorted(k for k in row if k < i):
            digest.update(struct.pack("<QQ", i, j))
    interior = branch_order(L)
    root = list(indices(L.root))
    return {
        "schema": "fan-symbolic-fill/1", "graph_sha256": digest.hexdigest(),
        "primal_vertices": L.n, "dynamics_dual_vertices": L.m,
        "kkt_lower_nnz_including_diagonal": len(pattern) + sum(map(len, pattern)) // 2,
        "ordering_algorithm": "exact minimum degree with vertex-index tie break",
        "branch_elimination": "reverse stage order, stage primal coordinates before incoming dynamics duals",
        "generic_minimum_degree": eliminate(pattern),
        "root_last_minimum_degree": eliminate(pattern, defer=root),
        "root_last": eliminate(pattern, interior + root),
        "root_first": eliminate(pattern, root + interior),
    }
