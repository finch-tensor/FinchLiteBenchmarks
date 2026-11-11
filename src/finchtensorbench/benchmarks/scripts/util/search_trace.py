"""Observe Finch's search decisions without replacing its search algorithms."""

import importlib
import inspect
import sys
from contextlib import contextmanager
from types import CodeType


def _order(fields):
    return " -> ".join(map(str, fields)) or "(empty)"


def _decision_lines(function):
    lines, first = inspect.getsourcelines(function)
    return {line.strip(): first + offset for offset, line in enumerate(lines)}


def _search_targets():
    logical = importlib.import_module(
        "finch.autoschedule.factorizer.galley_factorizer.branch_and_bound"
    )
    loops = importlib.import_module("finch.autoschedule.loop_orderer.loop_order_bnb")
    greedy = importlib.import_module("finch.autoschedule.loop_orderer.loop_order_greedy")
    functions = {
        "logical": logical.pruned_query_to_plan,
        "logical_bfs": logical.branch_and_bound,
        "logical_dfs": logical.branch_and_bound_dfs,
        "loop_bfs": loops.loop_order_bfs,
        "loop_dfs": loops.loop_order_dfs,
        "loop_greedy": greedy.greedy_loop_order,
    }
    targets = {
        function.__code__: (name, _decision_lines(function))
        for name, function in functions.items()
    }
    for code in greedy.greedy_loop_order.__code__.co_consts:
        if isinstance(code, CodeType) and code.co_name == "candidate_cost":
            targets[code] = ("loop_greedy_candidate", {})
    return targets, logical.GREEDY_COMPONENT_SIZE


@contextmanager
def capture_search(events):
    """Append (heading, text) snapshots of searches executed in this thread.

    Install this only around untimed warmup. Existing debuggers/coverage tracers
    continue receiving their events, and their tracer is restored on exit.
    Source markers keep the observations independent of upstream line numbers.
    """
    targets, greedy_threshold = _search_targets()
    previous = sys.gettrace()

    def record(frame, event, result, kind, lines):
        local = frame.f_locals
        line = frame.f_lineno
        if event == "call":
            if kind == "logical":
                events.append((
                    "Logical search configuration",
                    f"optimizer={local['optimizer']}; "
                    f"use_components={local['use_components']}; "
                    f"greedy fallback at {greedy_threshold} reduction indices",
                ))
            elif kind == "logical_bfs":
                mode = "greedy (k=1)" if local["k"] == 1 else "BFS branch-and-bound"
                events.append((
                    "Logical search started",
                    f"{mode}; component={_order(local['component'])}",
                ))
            elif kind == "logical_dfs":
                events.append((
                    "Logical search started",
                    f"DFS branch-and-bound; component={_order(local['component'])}",
                ))
            elif kind in ("loop_bfs", "loop_dfs", "loop_greedy"):
                mode = kind.removeprefix("loop_").upper()
                events.append((
                    "Loop-order search started",
                    f"{mode}; k={local.get('k', 'n/a')}; "
                    f"output indices={_order(local.get('output_vars') or ())}\n"
                    f"Expression: {local['expr']}",
                ))
            return

        if event == "return" and result is not None:
            if kind in ("logical_bfs", "logical_dfs"):
                state, _ = result
                events.append((
                    "Logical search selected",
                    f"order={_order(state[0])}; estimated cost={state[3]}",
                ))
            elif kind == "logical":
                events.append((
                    "Logical optimization completed",
                    f"order={_order(local['elimination_order'])}; "
                    f"estimated total cost={result[1]}",
                ))
            elif kind == "loop_greedy_candidate":
                events.append((
                    "Greedy loop candidate",
                    f"prefix={_order((*local['prefix'], local['field']))}; "
                    f"estimated incremental cost={result}",
                ))
            elif kind in ("loop_bfs", "loop_dfs", "loop_greedy"):
                cost = local.get("best_cost", "see candidate incremental costs")
                events.append((
                    "Loop-order search selected",
                    f"strategy={kind.removeprefix('loop_')}; "
                    f"order={_order(result)}; estimated cost={cost}",
                ))
            return

        if event != "line":
            return
        if kind == "logical" and line == lines.get(
            'if len(component) >= GREEDY_COMPONENT_SIZE or optimizer == "greedy":'
        ):
            component = local["component"]
            configured = local["optimizer"]
            if len(component) >= greedy_threshold:
                mode = "greedy fallback (component size threshold)"
            elif configured == "bfs":
                mode = "greedy bound followed by BFS branch-and-bound"
            else:
                mode = configured
            events.append((
                "Logical component strategy",
                f"component={_order(component)}; actual search={mode}",
            ))
        elif kind == "logical_bfs" and line == lines.get(
            "if total_cost <= cheapest_cost:"
        ):
            status = (
                "eligible for frontier"
                if local["total_cost"] <= local["cheapest_cost"]
                else "pruned: cost exceeds incumbent/bound"
            )
            events.append((
                "Logical reduction candidate",
                f"order={_order((*local['old_order'], local['idx']))}; "
                f"step cost={local['cost']}; total cost={local['total_cost']}; "
                f"incumbent/bound={local['cheapest_cost']}; {status}",
            ))
        elif kind == "logical_bfs" and line == lines.get(
            "new_optimal_orders = OrderedDict()"
        ):
            retained = local["top_k_idx_ext"]
            choices = []
            for key, (_, idx, order, _, cost) in local["sorted_items"]:
                status = "retained" if key in retained else "dropped by top-k limit"
                choices.append(f"{_order((*order, idx))}: cost={cost}; {status}")
            events.append(("Logical search frontier", "\n".join(choices)))
        elif kind == "logical_dfs" and line == lines.get("if new_cost > sup_best:"):
            if local["new_cost"] > local["sup_best"]:
                status = "pruned: exceeds upper bound"
            elif local["new_cost"] >= local["prev_best"]:
                status = "pruned: incumbent reaches this state at lower/equal cost"
            else:
                status = "eligible for frontier"
            events.append((
                "Logical DFS candidate",
                f"order={_order((*local['order'], local['idx']))}; "
                f"step cost={local['step_cost']}; total cost={local['new_cost']}; "
                f"upper bound={local['sup_best']}; "
                f"incumbent={local['prev_best']}; {status}",
            ))
        elif kind in ("loop_bfs", "loop_dfs") and line == lines.get(
            "if new_cost >= best_cost:"
        ):
            status = (
                "pruned: cost reaches/exceeds best complete order"
                if local["new_cost"] >= local["best_cost"]
                else "eligible for frontier or new complete order"
            )
            events.append((
                "Loop-order candidate",
                f"strategy={kind.removeprefix('loop_')}; "
                f"prefix={_order(local['new_prefix'])}; "
                f"estimated cumulative cost={local['new_cost']}; "
                f"best complete cost={local['best_cost']}; {status}",
            ))
        elif kind == "loop_greedy" and line == lines.get("prefix.append(best)"):
            events.append((
                "Greedy loop step selected",
                f"prefix={_order((*local['prefix'], local['best']))}",
            ))

    def dispatch(frame, event, arg):
        prior_local = previous(frame, event, arg) if previous is not None else None
        target = targets.get(frame.f_code)
        if target is None:
            return prior_local
        kind, lines = target
        record(frame, event, arg, kind, lines)

        def local_trace(frame, event, arg):
            nonlocal prior_local
            if prior_local is not None:
                prior_local = prior_local(frame, event, arg)
            record(frame, event, arg, kind, lines)
            return local_trace

        return local_trace

    try:
        sys.settrace(dispatch)
        yield
    finally:
        sys.settrace(previous)
