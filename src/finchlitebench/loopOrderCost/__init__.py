
"""Benchmarks comparing Finch's loop-order costs and execution times."""

from dataclasses import dataclass, field


@dataclass
class BenchmarkCase:
    name: str
    expression: object
    metadata: dict = field(default_factory=dict)
