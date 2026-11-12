"""An Array API namespace whose test arrays are sparse.

array-api-tests imports this module as its array namespace
(ARRAY_API_TESTS_MODULE=sparse_xp, plus ``-p sparse_xp``; see below), and
$SPARSE_XP chooses the library:

    finch   finch, inputs are FiberTensors (Dense levels over a SparseList,
            i.e. CSR for matrices), run with $FINCH_SCHEDULER
    torch   array_api_compat.torch, inputs are torch sparse COO tensors
    scorch  scorch, inputs are STensors. Scorch is a drop-in torch: names it
            does not define come from torch, here array_api_compat.torch, as
            for the torch row.

Each name is looked up on the library's own namespace. The only change is to
the arrays: the outputs of the spec's creation functions, and every array that
hypothesis draws for a test, are converted to the library's sparse format.

Building test inputs is scaffolding, not the library under test, so it runs
on dense arrays (and finch's default scheduler): hypothesis builds each array
densely (flat asarray, then reshape) and converts the finished one, and the
device and dtype probes array-api-tests runs while it is collected see dense
arrays. That is why this module is also a pytest plugin: its
pytest_collection_finish hook starts the sparse phase. 0-d arrays have no
sparse layout and stay dense.

pydata/sparse needs no adapter: its own creation functions return COO arrays.
"""

import contextlib
import functools
import os
import sys

LIBRARY = os.environ.get("SPARSE_XP", "")
start_tests = lambda: None  # noqa: E731  switch the library into test mode
dense_phase = contextlib.nullcontext  # context for building inputs

if LIBRARY == "finch":
    import numpy as np

    import finch as base
    from finch.algebra import ftype
    from finch.autoschedule import default_schedulers
    from finch.codegen import NumpyBuffer
    from finch.tensor import BufferizedNDArray, FiberTensor
    from finch.tensor.level import (
        DenseLevel,
        ElementLevel,
        SparseListLevel,
        dense,
        element,
        sparse_list,
    )

    # finch's default scheduler (INTERPRET_NOTATION) cannot evaluate
    # operations on FiberTensors, so tests run with $FINCH_SCHEDULER.
    _default_scheduler = default_schedulers.get_default_scheduler()
    _test_scheduler = getattr(
        default_schedulers, os.environ.get("FINCH_SCHEDULER", "INTERPRET_ASSEMBLY")
    )

    def start_tests():
        default_schedulers.set_default_scheduler(_test_scheduler)

    def dense_phase():
        return default_schedulers.with_default_scheduler(_default_scheduler)

    def to_sparse(x):
        # finch has no dense-to-sparse conversion yet, so the levels are built
        # here, in the layout FiberTensor.from_scipy_csr uses.
        if not (isinstance(x, BufferizedNDArray) and x.ndim >= 1):
            return x
        arr = np.asarray(x.to_numpy())
        rows = arr.reshape(-1, arr.shape[-1])
        mask = rows != 0  # NaN != 0, so NaNs are stored
        ptr = np.concatenate(([0], np.cumsum(mask.sum(axis=1)))).astype(np.intp)
        idx = np.nonzero(mask)[1].astype(np.intp)
        index_t = ftype(np.dtype(np.intp))
        element_t = element(
            fill_type=np.zeros((), dtype=arr.dtype)[()],
            element_type=ftype(arr.dtype),
            position_type=index_t,
        )
        lvl_t = sparse_list(element_t, dimension_type=index_t)
        lvl = SparseListLevel(
            ElementLevel(_format=element_t, _val=NumpyBuffer(rows[mask])),
            dimension=lvl_t.dimension_type(arr.shape[-1]),
            ptr=NumpyBuffer(ptr),
            idx=NumpyBuffer(idx),
        )
        for n in reversed(arr.shape[:-1]):
            lvl_t = dense(lvl_t, dimension_type=index_t)
            lvl = DenseLevel(lvl, dimension=lvl_t.dimension_type(n))
        return FiberTensor(lvl, _device=x.device)

elif LIBRARY in ("torch", "scorch"):
    import types

    import torch
    import array_api_compat.torch as base

    if LIBRARY == "scorch":
        import scorch

        class _Scorch:  # scorch's own definitions, then array_api_compat.torch
            def __getattr__(self, name, compat=base):
                own = vars(scorch).get(name)
                if own is None or name.startswith("__") or isinstance(own, types.ModuleType):
                    return getattr(compat, name)
                return own

        base = _Scorch()

    def to_sparse(x):
        if isinstance(x, torch.Tensor) and x.layout == torch.strided and x.ndim >= 1:
            x = x.to_sparse()  # COO over every dimension
            return scorch.STensor.from_torch(x) if LIBRARY == "scorch" else x
        return x

else:
    raise ImportError(f"set SPARSE_XP to finch, torch or scorch (got {LIBRARY!r})")


# The spec's creation functions (array_api_stubs/*/creation_functions.py).
CREATION_FUNCTIONS = [
    "arange", "asarray", "empty", "empty_like", "eye", "from_dlpack", "full",
    "full_like", "linspace", "meshgrid", "ones", "ones_like", "tril", "triu",
    "zeros", "zeros_like",
]
_dense = 1  # arrays stay dense while > 0: during collection and input building


def _sparsify(x):
    if _dense:
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(_sparsify(a) for a in x)
    return to_sparse(x)


def _wrap(func):
    @functools.wraps(func)  # keeps the library's signature for test_signatures
    def wrapper(*args, **kwargs):
        return _sparsify(func(*args, **kwargs))

    return wrapper


for _name in CREATION_FUNCTIONS:
    if hasattr(base, _name):  # names the library lacks stay missing
        setattr(sys.modules[__name__], _name, _wrap(getattr(base, _name)))


def __getattr__(name):
    return getattr(base, name)


def _patch_hypothesis():
    from hypothesis.extra import array_api

    draw = array_api.ArrayStrategy.do_draw

    def do_draw(self, data):
        global _dense
        _dense += 1
        try:
            with dense_phase():
                result = draw(self, data)
        finally:
            _dense -= 1
        return result if _dense else to_sparse(result)

    array_api.ArrayStrategy.do_draw = do_draw


_patch_hypothesis()


def pytest_collection_finish(session):
    """pytest hook (-p sparse_xp): collection is done, so the tests start."""
    global _dense
    _dense -= 1
    start_tests()
