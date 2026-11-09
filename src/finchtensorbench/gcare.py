"""Downloader and parsers for the G-CARE subgraph-matching benchmark.

Ported from the SparseArrayProgrammingSuite (``saps.downloaders.gcare``) so
this repository is self contained.  The data graph becomes one sparse 0/1
matrix per edge label (``E{label}``) and one sparse 0/1 vector per vertex label
(``V{label}``).  A query graph becomes one einsum: an index per query vertex,
an ``E`` operand per query edge, a ``V`` operand per labelled query vertex, and
a one-hot ``P{id}`` operand per query vertex pinned to a data vertex.  The
result is the scalar number of matches.

Reference:
    G-CARE: A Framework for Performance Benchmarking of Cardinality Estimation
    Techniques for Subgraph Matching -- Park et al., SIGMOD 2020.
    https://dl.acm.org/doi/10.1145/3318464.3389702
"""

from __future__ import annotations

import os
import string
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "gcare"

# (Google Drive link, sha256) for each archive, as pinned in SAPS.
_ARCHIVES = {
    "dataset": (
        "https://drive.google.com/file/d/1HAgSVE-24NOap6_Q1_twH56Dkb2kPvGU/view?usp=sharing",
        "78b86cda06115c4554cdfcfb93a7fbc8ecb759df39927510dd02ced4228a95e4",
    ),
    "queryset": (
        "https://drive.google.com/file/d/1Dlj43rBAOVPAsfzKlYxIbZ9RsqeGM_MN/view?usp=sharing",
        "c8dc9f978296559e9e55335a989ce16e7b5bcba7aa9d43e25fbd9e588d00ebc7",
    ),
    "ground_truth": (
        "https://drive.google.com/file/d/1Bc6Q2RZQTcIB8IfOw5KafNYwPhq2BO94/view?usp=sharing",
        "22e59f4fc06ffb79711d582513c6422ca555422c9947fde64a52f0a9292d382c",
    ),
}


def data_dir(root: str | Path | None = None) -> Path:
    """Resolve the G-CARE root: argument, then ``$GCARE_DATA_DIR``, then ``data/gcare``."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("GCARE_DATA_DIR") or DEFAULT_DATA_DIR)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def ensure_downloaded(root: str | Path | None = None) -> Path:
    """Download and extract the three G-CARE archives into *root* if absent."""
    import gdown
    from filelock import FileLock

    root = data_dir(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".download.lock")):
        for name, (link, sha256) in _ARCHIVES.items():
            target = root / name
            target.mkdir(parents=True, exist_ok=True)
            marker = target / f".extracted-{sha256}"
            if marker.exists():
                continue
            archive = target / f"{name}.tar.gz"
            gdown.cached_download(link, str(archive), hash=f"sha256:{sha256}")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=target, filter="data")
            marker.touch()
    return root


# ---------------------------------------------------------------------------
# Data graph
# ---------------------------------------------------------------------------


@dataclass
class GCareGraph:
    """A G-CARE data graph as COO index arrays (all stored values are 1).

    ``vectors[name]`` holds the vertex ids of a ``V{label}`` (or ``C``) vector;
    ``matrices[name]`` holds the ``(rows, cols)`` of an ``E{label}`` matrix.
    Every tensor has extent ``num_vertices = max_vid + 1`` along each mode.
    """

    name: str
    max_vid: int
    continuous_ids: bool
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def num_vertices(self) -> int:
        return self.max_vid + 1


def parse_graph(path: str | Path, name: str | None = None) -> GCareGraph:
    """Parse a G-CARE graph file (``t``/``v id labels...``/``e src dst labels...``).

    Unlabelled vertices and edges get label 0.  When vertex ids are not
    contiguous, an extra ``C`` vector marks the vertices that exist so that
    unlabelled query vertices cannot match the gaps.
    """
    path = Path(path)
    vertex_ids: list[int] = []
    by_vertex_label: dict[int, list[int]] = {}
    by_edge_label: dict[int, tuple[list[int], list[int]]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v"):
                vals = line.split()
                v_id = int(vals[1])
                vertex_ids.append(v_id)
                for label in [int(x) for x in vals[2:]] or [0]:
                    by_vertex_label.setdefault(label, []).append(v_id)
            elif line.startswith("e"):
                vals = line.split()
                src, dst = int(vals[1]), int(vals[2])
                for label in [int(x) for x in vals[3:]] or [0]:
                    rows, cols = by_edge_label.setdefault(label, ([], []))
                    rows.append(src)
                    cols.append(dst)

    max_vid = max(vertex_ids)
    graph = GCareGraph(
        name=name or path.stem,
        max_vid=max_vid,
        continuous_ids=max_vid + 1 == len(vertex_ids),
    )
    for label, verts in by_vertex_label.items():
        graph.vectors[f"V{label}"] = np.asarray(verts, dtype=np.int64)
    for label, (rows, cols) in by_edge_label.items():
        graph.matrices[f"E{label}"] = (
            np.asarray(rows, dtype=np.int64),
            np.asarray(cols, dtype=np.int64),
        )
    if not graph.continuous_ids:
        graph.vectors["C"] = np.asarray(vertex_ids, dtype=np.int64)
    return graph


def load_graph(dataset: str, root: str | Path | None = None) -> GCareGraph:
    if dataset in SGM_DATASETS:
        return parse_sgm_graph(ensure_sgm_downloaded(dataset, root) / "graph.txt", dataset)
    root = ensure_downloaded(root)
    return parse_graph(root / "dataset" / dataset / f"{dataset}.txt", name=dataset)


# ---------------------------------------------------------------------------
# Subgraph-matching benchmark graphs (as in Galley.jl's experiments)
# ---------------------------------------------------------------------------

# The in-memory subgraph matching benchmark of Sun & Luo (SIGMOD 2020) uses
# undirected graphs with one label per vertex and unlabelled edges.  Galley.jl's
# experiments use its yeast, dblp and youtube graphs with a "lite" sample of
# their queries; the files are fetched from that repository at a pinned commit.
_GALLEY_DATA = (
    "https://raw.githubusercontent.com/kylebd99/Galley.jl/"
    "e6fad3fcd7f1b1932b2529c89f59854f09a9a79b/Experiments/Data"
)
_LITE_IDS = [i for i in range(1, 201) if i % 10 in (0, 2, 4)]

# dataset -> (graph path, its sha256, query directory, query names)
SGM_DATASETS = {
    name: (
        f"Subgraph_Data/{name}/{name}.graph",
        sha256,
        f"{name}-lite",
        [f"{kind}/query_{kind}_{i}" for kind in ("dense_4", "sparse_8") for i in _LITE_IDS],
    )
    for name, sha256 in (
        ("yeast", "dd0119d77bcd95402da2fa8b2edc769de07c7d71837b2cd5fdeaa38c5b07302f"),
        ("dblp", "179ad985ed535a543bd98287f7e98122342789ae11648a830c46212a20de1ea0"),
        ("youtube", "473670a4dadb48aaaa973f392ea22cd3a1cf7bb9fb446bdd43795465d04084ea"),
    )
}


def ensure_sgm_downloaded(dataset: str, root: str | Path | None = None) -> Path:
    """Fetch *dataset*'s graph, queries, and true counts into ``<root>/sgm/<dataset>``.

    A query's true count lives at ``truth/<name>.txt``; it is absent for the
    few queries Galley.jl has none for.
    """
    import hashlib
    import urllib.error
    import urllib.request

    from filelock import FileLock

    graph_path, sha256, query_dir, names = SGM_DATASETS[dataset]
    target = data_dir(root) / "sgm" / dataset
    target.mkdir(parents=True, exist_ok=True)
    marker = target / f".complete-{sha256}"
    with FileLock(str(target / ".download.lock")):
        if marker.exists():
            return target

        def fetch(path: str) -> bytes | None:
            try:
                with urllib.request.urlopen(f"{_GALLEY_DATA}/{path}") as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                raise

        graph = fetch(graph_path)
        if graph is None or hashlib.sha256(graph).hexdigest() != sha256:
            raise RuntimeError(f"bad download of {graph_path}")
        (target / "graph.txt").write_bytes(graph)
        for name in names:
            stem = name.split("/")[1]
            for kind, folder in (("queries", f"Subgraph_Queries/{query_dir}"),
                                 ("truth", f"Subgraph_TrueCardinalities/{dataset}")):
                data = fetch(f"{folder}/{stem}.graph")
                if data is None and kind == "queries":
                    raise RuntimeError(f"missing query {name}")
                if data is not None:
                    path = target / kind / f"{name}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
        marker.touch()
    return target


def parse_sgm_graph(path: str | Path, name: str) -> GCareGraph:
    """Parse a ``t n m``/``v id label degree``/``e u v label`` graph file.

    Edges are undirected, so ``E0`` holds both directions of each.
    """
    vertex_ids: list[int] = []
    labels: list[int] = []
    edges: list[tuple[int, int]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v"):
                vals = line.split()
                vertex_ids.append(int(vals[1]))
                labels.append(int(vals[2]))
            elif line.startswith("e"):
                vals = line.split()
                edges.append((int(vals[1]), int(vals[2])))
    ids = np.asarray(vertex_ids, dtype=np.int64)
    label_of = np.asarray(labels, dtype=np.int64)
    max_vid = int(ids.max())
    graph = GCareGraph(name=name, max_vid=max_vid, continuous_ids=max_vid + 1 == len(ids))
    for label in np.unique(label_of):
        graph.vectors[f"V{label}"] = np.sort(ids[label_of == label])
    pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    both = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    graph.matrices["E0"] = (both[:, 0].copy(), both[:, 1].copy())
    if not graph.continuous_ids:
        graph.vectors["C"] = np.sort(ids)
    return graph


def parse_sgm_query(
    path: str | Path,
    *,
    continuous_ids: bool = True,
    dataset: str = "",
    name: str = "",
    ground_truth: int | None = None,
) -> GCareQuery:
    """Parse a ``v qid label degree``/``e q1 q2 label`` query: every vertex is
    labelled and every edge reads ``E0``."""
    vertices: list[int] = []
    operands: list[Operand] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            vals = line.split()
            if line.startswith("v"):
                qv = int(vals[1])
                vertices.append(qv)
                operands.append(Operand(f"V{int(vals[2])}", (qv,)))
                if not continuous_ids:
                    operands.append(Operand("C", (qv,)))
            elif line.startswith("e"):
                operands.append(Operand("E0", (int(vals[1]), int(vals[2]))))
    return GCareQuery(
        dataset=dataset,
        name=name,
        vertices=tuple(vertices),
        operands=tuple(operands),
        ground_truth=ground_truth,
    )


# ---------------------------------------------------------------------------
# Query graphs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operand:
    """One einsum operand: a graph tensor name and the query vertices it binds."""

    tensor: str
    vertices: tuple[int, ...]


@dataclass(frozen=True)
class GCareQuery:
    """A parsed query graph, ready to be lowered to an einsum."""

    dataset: str
    name: str  # path under ``queryset/<dataset>/`` without ``.txt``
    vertices: tuple[int, ...]
    operands: tuple[Operand, ...]
    ground_truth: int | None = None

    @property
    def family(self) -> str:
        return self.name.split("/")[0]

    @property
    def tensor_names(self) -> list[str]:
        """Distinct tensors the query reads, sorted (one argument each)."""
        return sorted({op.tensor for op in self.operands})

    def saps_expr(self) -> str:
        """The SAPS/einop spelling, e.g. ``S[] += V3[v_0] * E0[v_0,v_1]``."""
        terms = [
            f"{op.tensor}[{','.join(f'v_{v}' for v in op.vertices)}]"
            for op in self.operands
        ]
        return "S[] += " + " * ".join(terms)

    def subscripts(self) -> str:
        """NumPy-style subscripts, one letter per query vertex, e.g. ``a,ab->``."""
        letters = string.ascii_letters
        if len(self.vertices) > len(letters):
            raise ValueError(f"{self.name} has too many vertices for einsum letters")
        letter = {v: letters[i] for i, v in enumerate(self.vertices)}
        return (
            ",".join("".join(letter[v] for v in op.vertices) for op in self.operands)
            + "->"
        )


def parse_query(
    path: str | Path,
    *,
    continuous_ids: bool = True,
    dataset: str = "",
    name: str = "",
    ground_truth: int | None = None,
) -> GCareQuery:
    """Parse a G-CARE query file.

    ``v qid label data_id``: label -1 means any label; data_id -1 means the
    vertex is free, otherwise it is pinned to that data vertex (``P{id}``).
    ``e qid1 qid2 label``: a directed, labelled edge (``E{label}[q1, q2]``).
    """
    vertices: list[int] = []
    operands: list[Operand] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v"):
                vals = line.split()
                qv, label, data_id = int(vals[1]), int(vals[2]), int(vals[3])
                vertices.append(qv)
                if label != -1:
                    operands.append(Operand(f"V{label}", (qv,)))
                if data_id != -1:
                    operands.append(Operand(f"P{data_id}", (qv,)))
                elif not continuous_ids:
                    operands.append(Operand("C", (qv,)))
            elif line.startswith("e"):
                vals = line.split()
                operands.append(
                    Operand(f"E{int(vals[3])}", (int(vals[1]), int(vals[2])))
                )
    return GCareQuery(
        dataset=dataset,
        name=name,
        vertices=tuple(vertices),
        operands=tuple(operands),
        ground_truth=ground_truth,
    )


def _read_ground_truth(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(path.read_text().split()[0])


def list_queries(dataset: str, root: str | Path | None = None) -> list[str]:
    """Query names (e.g. ``Star_3/uf_Q_0_1``) available for *dataset*, sorted."""
    if dataset in SGM_DATASETS:
        return sorted(SGM_DATASETS[dataset][3])
    root = ensure_downloaded(root)
    base = root / "queryset" / dataset
    return sorted(p.relative_to(base).with_suffix("").as_posix() for p in base.rglob("*.txt"))


def load_queries(
    graph: GCareGraph,
    names: list[str] | None = None,
    root: str | Path | None = None,
) -> list[GCareQuery]:
    """Parse the queries of *graph*'s dataset, with their ground-truth counts.

    Ground truth is matched by the query's relative path.  (SAPS matched by
    file stem, but stems such as ``uf_Q_1_1`` repeat across query families.)
    """
    if graph.name in SGM_DATASETS:
        root = ensure_sgm_downloaded(graph.name, root)
        query_root, truth_root, parse = root / "queries", root / "truth", parse_sgm_query
    else:
        root = ensure_downloaded(root)
        query_root = root / "queryset" / graph.name
        truth_root = root / "ground_truth" / graph.name
        parse = parse_query
    names = list_queries(graph.name, root) if names is None else names
    return [
        parse(
            query_root / f"{name}.txt",
            continuous_ids=graph.continuous_ids,
            dataset=graph.name,
            name=name,
            ground_truth=_read_ground_truth(truth_root / f"{name}.txt"),
        )
        for name in names
    ]


def query_tensor_coo(
    graph: GCareGraph, tensor: str
) -> tuple[int, tuple[np.ndarray, ...]]:
    """COO coordinates for one operand of a query, as ``(ndim, coords)``.

    Labels absent from the data graph become empty tensors; ``P{id}`` becomes a
    one-hot vector at data vertex ``id``.
    """
    if tensor in graph.matrices:
        return 2, graph.matrices[tensor]
    if tensor in graph.vectors:
        return 1, (graph.vectors[tensor],)
    empty = np.zeros(0, dtype=np.int64)
    if tensor.startswith("P"):
        return 1, (np.asarray([int(tensor[1:])], dtype=np.int64),)
    if tensor.startswith("V") or tensor == "C":
        return 1, (empty,)
    if tensor.startswith("E"):
        return 2, (empty, empty)
    raise ValueError(f"unknown G-CARE tensor name {tensor!r}")
