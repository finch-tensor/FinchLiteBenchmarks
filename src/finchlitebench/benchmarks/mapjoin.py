from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

import finchlite as fl
from finchlite import ffuncs
from finchlite.autoschedule import Aggregate, Field, Literal, MapJoin, Table
from finchlite.autoschedule.tensor_stats import StatsInterpreter


# A_ij * B_ij
def benchmark_elementwise_mul(stats_factory, A, B):
    i, j = Field("i"), Field("j")
    fl_A, fl_B = fl.asarray(A), fl.asarray(B)
    node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (i, j))),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


# A_ij + B_ij
def benchmark_elementwise_add(stats_factory, A, B):
    i, j = Field("i"), Field("j")
    A_, B_ = fl.asarray(A), fl.asarray(B)
    node = MapJoin(
        Literal(ffuncs.add),
        (Table(Literal(A_), (i, j)), Table(Literal(B_), (i, j))),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


# A_ij * b_j
def benchmark_broadcast_mul(stats_factory, A, b):
    i, j = Field("i"), Field("j")
    fl_A, fl_b = fl.asarray(A), fl.asarray(b)
    node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(fl_A), (i, j)), Table(Literal(fl_b), (j,))),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


# A_ij + b_j
def benchmark_broadcast_add(stats_factory, A, b):
    i, j = Field("i"), Field("j")
    fl_A, fl_b = fl.asarray(A), fl.asarray(b)
    node = MapJoin(
        Literal(ffuncs.add),
        (Table(Literal(fl_A), (i, j)), Table(Literal(fl_b), (j,))),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


# A_ij * B_jk = C_ijk
def benchmark_join_mul(stats_factory, A, B):
    i, j, k = Field("i"), Field("j"), Field("k")
    fl_A, fl_B = fl.asarray(A), fl.asarray(B)
    node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (j, k))),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


# A_ij + B_jk = C_ijk
def benchmark_join_add(stats_factory, A, B):
    i, j, k = Field("i"), Field("j"), Field("k")
    fl_A, fl_B = fl.asarray(A), fl.asarray(B)
    node = MapJoin(
        Literal(ffuncs.add),
        (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (j, k))),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


def benchmark_semiring(stats_factory, A, B, C, D):
    i, j, k, l = Field("i"), Field("j"), Field("k"), Field("l")
    A_, B_, C_, D_ = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)
    ab = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(Literal(A_), (i, j)), Table(Literal(B_), (j, k)))),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(Literal(ffuncs.mul), (Table(Literal(C_), (i, l)), Table(Literal(D_), (l, k)))),
        (l,),
    )
    node = MapJoin(Literal(ffuncs.mul), (ab, cd))
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


def benchmark_semiring_tropical(stats_factory, A, B, C, D):
    i, j, k, l = Field("i"), Field("j"), Field("k"), Field("l")
    fl_A, fl_B, fl_C, fl_D = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)
    ab = Aggregate(
        Literal(ffuncs.min),
        Literal(float("inf")),
        MapJoin(Literal(ffuncs.add), (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (j, k)))),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.min),
        Literal(float("inf")),
        MapJoin(Literal(ffuncs.add), (Table(Literal(fl_C), (i, l)), Table(Literal(fl_D), (l, k)))),
        (l,),
    )
    node = MapJoin(Literal(ffuncs.add), (ab, cd))
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


def benchmark_semiring_maxplus(stats_factory, A, B, C, D):
    i, j, k, l = Field("i"), Field("j"), Field("k"), Field("l")
    fl_A, fl_B, fl_C, fl_D = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)
    ab = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(Literal(ffuncs.add), (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (j, k)))),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(Literal(ffuncs.add), (Table(Literal(fl_C), (i, l)), Table(Literal(fl_D), (l, k)))),
        (l,),
    )
    node = MapJoin(Literal(ffuncs.add), (ab, cd))
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


def benchmark_semiring_boolean(stats_factory, A, B, C, D):
    i, j, k, l = Field("i"), Field("j"), Field("k"), Field("l")
    fl_A, fl_B, fl_C, fl_D = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)
    ab = Aggregate(
        Literal(ffuncs.or_),
        Literal(False),
        MapJoin(Literal(ffuncs.and_), (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (j, k)))),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.or_),
        Literal(False),
        MapJoin(Literal(ffuncs.and_), (Table(Literal(fl_C), (i, l)), Table(Literal(fl_D), (l, k)))),
        (l,),
    )
    node = MapJoin(Literal(ffuncs.and_), (ab, cd))
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


def benchmark_semiring_maxmin(stats_factory, A, B, C, D):
    i, j, k, l = Field("i"), Field("j"), Field("k"), Field("l")
    fl_A, fl_B, fl_C, fl_D = fl.asarray(A), fl.asarray(B), fl.asarray(C), fl.asarray(D)
    ab = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(Literal(ffuncs.min), (Table(Literal(fl_A), (i, j)), Table(Literal(fl_B), (j, k)))),
        (j,),
    )
    cd = Aggregate(
        Literal(ffuncs.max),
        Literal(float("-inf")),
        MapJoin(Literal(ffuncs.min), (Table(Literal(fl_C), (i, l)), Table(Literal(fl_D), (l, k)))),
        (l,),
    )
    node = MapJoin(Literal(ffuncs.min), (ab, cd))
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()