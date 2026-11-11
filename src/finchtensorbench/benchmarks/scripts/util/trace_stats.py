"""Read-only formatting of statistics and configuration used by Finch stages."""

import math

from finch.autoschedule.cost_models import FLOPS_COST_MODEL
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)


def _number(value):
    try:
        return f"{value:.12g}"
    except (TypeError, ValueError):
        return str(value)


def _fields(fields):
    return "{" + ", ".join(sorted(map(str, fields))) + "}"


def _stat_layers(stats):
    """Unwrap MemoStats without estimating sizes or filling its caches."""
    seen = set()
    while stats is not None and id(stats) not in seen:
        seen.add(id(stats))
        yield stats
        stats = getattr(stats, "inner", None)


def format_stats(stats_bindings):
    """Describe available bounds; never scan tensors or invoke estimators."""
    lines = []
    for alias, stats in stats_bindings.items():
        order = tuple(getattr(stats, "index_order", ()))
        dimensions = getattr(stats, "dim_sizes", {})
        sizes = ", ".join(f"{field}={_number(size)}" for field, size in dimensions.items())
        lines.append(f"{alias}: {type(stats).__name__}, index_order=({', '.join(map(str, order))})")
        lines.append(f"  dimensions: {sizes or '(scalar)'}; fill_type={getattr(stats, 'fill_type', 'unknown')}")

        layers = list(_stat_layers(stats))
        constraints = next((list(layer.dcs) for layer in layers if hasattr(layer, "dcs")), [])
        capacity = math.prod(dimensions.values())
        bounds = [capacity]
        axes = frozenset(order)
        for constraint in constraints:
            if not constraint.from_indices and axes <= constraint.to_indices:
                bounds.append(constraint.value)
        if dimensions or not order:
            lines.append(f"  non-fill cardinality upper bound <= {_number(min(bounds))} (dense capacity and direct constraints)")

        for layer in layers:
            estimates = getattr(getattr(layer, "node", None), "estimates", {})
            for fixed, value in sorted(estimates.items(), key=lambda item: tuple(sorted(map(str, item[0])))):
                target = "non-fill cardinality" if not fixed else f"largest slice fixing {_fields(fixed)}"
                lines.append(f"  cached {target} estimate: {_number(value)}")

        constraints.sort(key=lambda dc: (tuple(sorted(map(str, dc.from_indices))),
                                         tuple(sorted(map(str, dc.to_indices))), dc.p))
        for constraint in constraints:
            source, target = constraint.from_indices, constraint.to_indices
            if not source:
                label = f"distinct support tuples on {_fields(target)}"
            elif math.isinf(constraint.p):
                label = f"max degree {_fields(target)} given {_fields(source)}"
            else:
                label = f"degree L{_number(constraint.p)} norm {_fields(target)} given {_fields(source)}"
            lines.append(f"  {label} <= {_number(constraint.value)}")
    return "\n".join(lines) or "(no statistics available)"


def _describe(value):
    if value is None:
        return "None (upstream default)"
    if isinstance(value, (str, bool, int, float)):
        return repr(value)
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    name = type(value).__name__
    inner = getattr(value, "inner", None)
    return f"{name}({_describe(inner)})" if inner is not None else name


def _cost_model(model):
    basis = getattr(model, "basis", None)
    coefficients = getattr(model, "coefficients", None)
    names = getattr(basis, "feature_names", None)
    if names is None or coefficients is None:
        return _describe(model)
    terms = [f"{_number(coefficient)} * {name}"
             for name, coefficient in zip(names, coefficients, strict=True)]
    return f"{type(model).__name__}, basis={type(basis).__name__}; cost = " + (" + ".join(terms) or "0")


def format_config(stage):
    """Describe stored stage options without running the cost model."""
    lines = []
    for field in ("optimizer", "use_components", "k", "stats_factory", "cost_model", "format_cost", "cache"):
        if hasattr(stage, field):
            value = getattr(stage, field)
            if field == "cost_model" and value is None and isinstance(stage, GalleyLogicFactorizer):
                description = "FLOPS_COST_MODEL (upstream default): " + _cost_model(FLOPS_COST_MODEL)
            else:
                description = _cost_model(value) if field == "cost_model" else _describe(value)
            lines.append(f"{field}={description}")
    return "\n".join(lines) or "(no configurable options recorded)"
