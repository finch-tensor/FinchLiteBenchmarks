"""SpGEMM inputs from the Birdseed paper artifact (outer_transpose).

make_inputs(shared) regenerates the artifact's structure for any number of shared
active columns; load_inputs() still loads the exact committed files (shared = 64)."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import scipy.sparse as sps
from scipy.io import mmread


ROWS = 4_194_304
COLS = 131_072
ACTIVE_COLUMNS = 65_536
NONZEROS_PER_COLUMN = 16
SHARED_COLUMNS = 64  # the committed artifact files
SEED = 42
ARTIFACT_REVISION = "32a89f4be661f609eea0cd2780689cfc1677aa5e"
SOURCE = (
    "https://github.com/finch-tensor/FinchBenchmarks/blob/"
    f"{ARTIFACT_REVISION}/outer_transpose/make_outer_data.jl"
)
INPUTS = {
    "outer_A.ttx": "46d26a77c2ab610d17b1866509248b2cb3ebc9fe",
    "outer_B.ttx": "21f3bc4f66ed606c6f58bed2e5b1230ce095beba",
}
DATA_DIR = Path(__file__).resolve().parents[5] / ".cache" / "outer_transpose"


def _git_blob_hash(path):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_path(name, expected_hash, data_dir):
    path = data_dir / name
    if not path.is_file():
        url = (
            "https://raw.githubusercontent.com/finch-tensor/FinchBenchmarks/"
            f"{ARTIFACT_REVISION}/outer_transpose/data/{name}"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        print(f"Downloading original paper input: {url}", flush=True)
        try:
            with urlopen(url, timeout=120) as source, temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            if _git_blob_hash(temporary) != expected_hash:
                raise ValueError(f"Downloaded paper input failed its hash check: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if _git_blob_hash(path) != expected_hash:
        raise ValueError(f"Cached paper input failed its hash check: {path}")
    return path


def validate_inputs(a, b, shared_columns=SHARED_COLUMNS):
    """Check the paper's shape, degrees, cardinalities, and support overlap."""
    supports = []
    for name, matrix in (("A", a), ("B", b)):
        counts = np.diff(matrix.indptr)
        if matrix.shape != (ROWS, COLS) or matrix.nnz != ACTIVE_COLUMNS * NONZEROS_PER_COLUMN:
            raise ValueError(f"Paper input {name} has an unexpected shape or nonzero count")
        if not matrix.has_canonical_format or not np.all((counts == 0) | (counts == NONZEROS_PER_COLUMN)):
            raise ValueError(f"Paper input {name} does not have 16 distinct entries per active column")
        support = np.flatnonzero(counts)
        if support.size != ACTIVE_COLUMNS:
            raise ValueError(f"Paper input {name} does not have {ACTIVE_COLUMNS} active columns")
        supports.append(support)
    shared = np.intersect1d(*supports, assume_unique=True)
    if shared.size != shared_columns:
        raise ValueError(f"Paper inputs do not share exactly {shared_columns} active columns")


def load_inputs(data_dir=DATA_DIR):
    """Preserve the artifact's original coordinates and uniform random values."""
    matrices = [
        sps.csc_array(mmread(_input_path(name, expected, Path(data_dir))))
        for name, expected in INPUTS.items()
    ]
    validate_inputs(*matrices)
    for matrix in matrices:
        matrix.indices = matrix.indices.astype(np.intp, copy=False)
        matrix.indptr = matrix.indptr.astype(np.intp, copy=False)
    return tuple(matrices)


def _random_columns(rng, columns):
    """CSC (ROWS x COLS) with NONZEROS_PER_COLUMN distinct random rows in each given column."""
    columns = np.sort(columns)
    rows = rng.integers(0, ROWS, size=(columns.size, NONZEROS_PER_COLUMN))
    rows.sort(axis=1)
    while (repeat := np.flatnonzero((np.diff(rows, axis=1) == 0).any(axis=1))).size:
        rows[repeat] = np.sort(rng.integers(0, ROWS, size=(repeat.size, NONZEROS_PER_COLUMN)), axis=1)
    counts = np.zeros(COLS, dtype=np.intp)
    counts[columns] = NONZEROS_PER_COLUMN
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.intp)
    indices = rows.reshape(-1).astype(np.intp)
    return sps.csc_array((rng.random(indices.size), indices, indptr), shape=(ROWS, COLS))


def make_inputs(shared_columns, seed=SEED):
    """Same construction as the artifact's make_outer_data.jl, for any shared count:
    shuffle the columns, give `shared_columns` of them to both A and B, and the rest
    of each matrix's ACTIVE_COLUMNS to that matrix alone."""
    if not 0 <= shared_columns <= ACTIVE_COLUMNS or 2 * ACTIVE_COLUMNS - shared_columns > COLS:
        raise ValueError(f"shared_columns must be between 0 and {ACTIVE_COLUMNS}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(COLS)
    shared = order[:shared_columns]
    only_a = order[shared_columns:ACTIVE_COLUMNS]
    only_b = order[ACTIVE_COLUMNS:2 * ACTIVE_COLUMNS - shared_columns]
    a = _random_columns(rng, np.concatenate((shared, only_a)))
    b = _random_columns(rng, np.concatenate((shared, only_b)))
    validate_inputs(a, b, shared_columns)
    return a, b
