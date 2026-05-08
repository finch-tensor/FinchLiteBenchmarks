from __future__ import annotations

import numpy as np

import finchlite as fl
import finchlite.interface.eager as eager


class PureStats:
    def __init__(self, *matrices: np.ndarray):
        result = fl.asarray(matrices[0])
        for m in matrices[1:]:
            result = eager.matmul(result, fl.asarray(m))
        self.eager_nnz = float(np.count_nonzero(result.to_numpy()))
