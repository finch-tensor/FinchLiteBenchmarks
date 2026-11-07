import csv
import itertools
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sps

import finch as fl
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.executor import LogicExecutor
from finch.autoschedule.factorizer.galley_factorizer import insert_statistics
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.factorizer.optimize import DefaultLogicFactorizer
from finch.autoschedule.formatter import GalleyFormatter
from finch.autoschedule.loop_orderer import BFSLoopOrderer
from finch.autoschedule.loop_orderer.loop_order_cost import (
    cost_of_reformat,
    loop_order_cost,
)
from finch.autoschedule.loop_orderer.loop_ordering import (
    AbstractLoopOrderer,
)
from finch.autoschedule.normalize import LogicNormalizer
from finch.autoschedule.tensor_stats import UniformStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import (
    Aggregate,
    Alias,
    LogicSimplify,
    Plan,
    Query,
    Reorder,
    Table,
)
from finch.symbolic import PostOrderDFS, gensym

SIZE = 2048  # M = N
K_EXPS = range(11, 12)  # K = 2^e
DENSITY_EXPS = range(1, 23)  # density = 2^e / (M * N)
Q_EXPS = (5, 11, 17)  # B fixed at nnz 2^5, 2^11, 2^17
WARMUP = 1
RUNS = 3
OUT = Path("sddmm_microbenchmark.csv")

LOOP_ORDERS = ("ijk", "ikj", "jik", "jki", "kij", "kji")
FIELDS = (
    "M",
    "N",
    "K",
    "p",
    "q",
    "r",
    "nnz_A",
    "nnz_B",
    "nnz_C",
    "variant",
    "orderer",
    "executed_order",
    "est_cost",
    "fused_plan",
    "seconds",
    "plan_seconds",
    "bruteforce_order",
    "matches_bruteforce",
)


def fused_sddmm(ctx, A, B, C):
    return fl.compute(fl.einsum("ij,jk,ik->ik", A, B, C), ctx=ctx)


class FixedLoopOrderer(AbstractLoopOrderer):
    """Force a permutation of output, reduction, and output index names."""

    def __init__(self, ctx, order):
        super().__init__(ctx)
        self.order = order

    def set_loop_orders(self, prgm, stats, stats_factory):
        queries = []
        for query in prgm.bodies[:-1]:
            match query:
                case Query(Table(_, out) as lhs, Aggregate(op, init, arg, red)):
                    role = dict(
                        zip(sorted(self.order), (out[0], *red, out[1]), strict=True)
                    )
                    self.ctx.roles = {field: name for name, field in role.items()}
                    order = tuple(role[c] for c in self.order)
                    output_order = tuple(f for f in order if f in out)
                    if output_order != out:
                        # Finch needs output storage to follow the loop order.
                        tmp = Table(Alias(gensym("sddmm")), output_order)
                        queries.append(
                            Query(tmp, Aggregate(op, init, Reorder(arg, order), red))
                        )
                        queries.append(Query(lhs, tmp))
                        continue
                    query = Query(lhs, Aggregate(op, init, Reorder(arg, order), red))
            queries.append(query)
        return Plan((*queries, prgm.bodies[-1]))


class Recorder:
    """Pipeline stage between the loop orderer and the backend.

    Records the executed loop orders, their estimated cost, whether the plan is
    fused, and the kernel runtime. Compiles each distinct program once.
    """

    _GENSYM = re.compile(r"#\w+#\d+|\b[A-Za-z]\w*?_\d+\b")

    def __init__(self, backend):
        self.backend, self.kernels = backend, {}
        self.reset()

    def reset(self):
        self.orders, self.cost, self.fused, self.seconds = [], 0.0, False, 0.0
        self.roles = {}

    def __call__(self, prgm, bindings, stats, stats_factory):
        stats_bindings, cache = dict(stats), {}
        for query in prgm.bodies[:-1]:
            if isinstance(query.rhs, Table):
                self.cost += cost_of_reformat(stats_bindings[query.rhs.tns])
            for node in PostOrderDFS(query.rhs):
                if isinstance(node, Aggregate) and isinstance(node.arg, Reorder):
                    order, out = node.arg.idxs, query.lhs.idxs
                    role = {out[0]: "i", out[-1]: "k"} | dict.fromkeys(node.idxs, "j")
                    role.update(self.roles)
                    self.orders.append("".join(role.get(f, "?") for f in order))
                    self.cost += loop_order_cost(
                        node.arg.arg, order, stats_factory, stats_bindings, out
                    )
                    tables = {
                        t.tns for t in PostOrderDFS(node.arg) if isinstance(t, Table)
                    }
                    self.fused |= bool(node.idxs) and len(tables) >= 3
            insert_statistics(
                stats_factory, query, stats_bindings, replace=False, cache=cache
            )

        names = {}
        text = self._GENSYM.sub(
            lambda m: names.setdefault(m.group(), f"v{len(names)}"), str(prgm)
        )
        # Galley's storage choices depend on input sizes and densities.
        key = (
            text,
            tuple(
                (
                    repr(t),
                    tuple(stats[var].dim_sizes.values()),
                    stats[var].estimate_non_fill_values(),
                )
                for var, t in bindings.items()
            ),
        )
        if key not in self.kernels:
            mod, *rest = self.backend(prgm, bindings, stats, stats_factory)

            def main(*args, mod=mod):
                start = time.perf_counter()
                result = mod.main(*args)
                self.seconds += time.perf_counter() - start
                return result

            self.kernels[key] = (SimpleNamespace(main=main), *rest)
        return self.kernels[key]


RECORDER = Recorder(
    GalleyFormatter(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())))
)


def scheduler(orderer, galley=False):
    factorizer = GalleyLogicFactorizer if galley else DefaultLogicFactorizer
    # No plan cache: each call is planned with the current inputs' statistics.
    return LogicNormalizer(
        LogicExecutor(
            factorizer(LogicSimplify(orderer)),
            stats_factory=UniformStatsFactory(),
        )
    )


# Julia Galley stages, with uniform statistics for this benchmark.
SCHEDULERS = [
    ("forced", o, scheduler(FixedLoopOrderer(RECORDER, o))) for o in LOOP_ORDERS
] + [("autoscheduled", "bfs", scheduler(BFSLoopOrderer(RECORDER), True))]


def measure(ctx, inputs):
    for _ in range(WARMUP):
        print("    warmup", flush=True)
        fused_sddmm(ctx, *inputs)
    kernel, wall = [], []
    for run in range(1, RUNS + 1):
        print(f"    timed run {run}/{RUNS}", flush=True)
        RECORDER.reset()
        start = time.perf_counter()
        fused_sddmm(ctx, *inputs)
        wall.append(time.perf_counter() - start)
        kernel.append(RECORDER.seconds)
    return {
        "executed_order": " | ".join(RECORDER.orders),
        "est_cost": RECORDER.cost,
        "fused_plan": RECORDER.fused,
        "seconds": float(np.mean(kernel)),
        "plan_seconds": float(
            np.mean([w - k for w, k in zip(wall, kernel, strict=True)])
        ),
    }


def main():
    M = N = SIZE
    configs = list(itertools.product(K_EXPS, DENSITY_EXPS, Q_EXPS, DENSITY_EXPS))
    print(f"M=N={M}: {len(configs)} configurations, {len(SCHEDULERS)} schedules each")

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for config_id, (k_exp, pe, qe, rexp) in enumerate(configs, 1):
            K = 2**k_exp
            p, q, r = (2**e / (M * N) for e in (pe, qe, rexp))
            rng = np.random.default_rng([k_exp, pe, qe, rexp])
            mats = [
                sps.random(
                    m,
                    n,
                    density=min(1.0, d),
                    format="csr",
                    random_state=rng,
                    data_rvs=np.ones,
                    dtype=np.float64,
                )
                for (m, n), d in (((M, N), p), ((N, K), q), ((M, K), r))
            ]
            # Match Julia's integer writes to the CSR index-buffer type.
            for m in mats:
                m.indices = m.indices.astype(np.int64, copy=False)
                m.indptr = m.indptr.astype(np.int64, copy=False)
            inputs = [fl.defer(fl.asarray(m)) for m in mats]
            base = {
                "M": M,
                "N": N,
                "K": K,
                "p": p,
                "q": q,
                "r": r,
                "nnz_A": mats[0].nnz,
                "nnz_B": mats[1].nnz,
                "nnz_C": mats[2].nnz,
            }

            print(
                f"[{config_id}/{len(configs)}] K={K} p=2^{pe} q=2^{qe} r=2^{rexp}:",
                flush=True,
            )
            rows = []
            for variant, name, ctx in SCHEDULERS:
                print(f"  measuring {variant} {name}", flush=True)
                row = base | {"variant": variant, "orderer": name}
                row |= measure(ctx, inputs)
                if variant == "forced" and (
                    row["executed_order"] != name or not row["fused_plan"]
                ):
                    raise RuntimeError(
                        f"Expected fused {name}, got {row['executed_order']}"
                    )
                rows.append(row)
                print(
                    f"  {variant:13s} {name:8s} ran {row['executed_order']:5s} "
                    f"{row['seconds'] * 1e3:9.3f} ms  cost {row['est_cost']:.3g}"
                )

            # Bruteforce picks the cheapest order under the cost model.
            best = min(
                (row for row in rows if row["variant"] == "forced"),
                key=lambda row: row["est_cost"],
            )["executed_order"]
            for row in rows:
                row["bruteforce_order"] = best
                row["matches_bruteforce"] = row["executed_order"] == best
                writer.writerow(row)
            print(
                f"  bruteforce picks {best}; autoscheduled "
                + ("matches" if rows[-1]["matches_bruteforce"] else "differs")
            )
            f.flush()

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
