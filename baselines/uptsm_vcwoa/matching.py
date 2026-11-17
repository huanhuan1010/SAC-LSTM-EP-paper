from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy rectangular Hungarian solver for finite minimization costs.

    This avoids making SciPy a mandatory dependency on the legacy server.
    The implementation follows the shortest augmenting path formulation and
    returns min(rows, columns) one-to-one assignments.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2 or cost.size == 0:
        raise ValueError("cost_matrix must be a non-empty 2-D array")
    if not np.all(np.isfinite(cost)):
        raise ValueError("cost_matrix must contain only finite values")

    transposed = cost.shape[0] > cost.shape[1]
    work = cost.T.copy() if transposed else cost.copy()
    n_rows, n_cols = work.shape

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for row in range(1, n_rows + 1):
        p[0] = row
        min_value = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, n_cols + 1):
                if used[column]:
                    continue
                current = work[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(n_cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    pairs = [(int(p[column] - 1), column - 1)
             for column in range(1, n_cols + 1) if p[column] != 0]
    pairs.sort()
    rows = np.array([pair[0] for pair in pairs], dtype=np.int64)
    cols = np.array([pair[1] for pair in pairs], dtype=np.int64)
    if transposed:
        return cols, rows
    return rows, cols


def maximum_weight_uav_matching(
    edge_weights_ue_uav: np.ndarray,
    allow_idle: bool = True,
) -> list[tuple[int, int]]:
    """Return (user, UAV) pairs for P3, optionally with zero-weight idling."""
    weights = np.asarray(edge_weights_ue_uav, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("edge weight matrix must be 2-D")
    num_users, num_uavs = weights.shape
    # Rows are UAVs and columns are UEs. Dummy UEs let a UAV remain idle when
    # every real edge has negative profit, matching the <= constraints in P3.
    transposed = weights.T
    if allow_idle:
        transposed = np.concatenate(
            [transposed, np.zeros((num_uavs, num_uavs))], axis=1
        )
    rows, cols = linear_sum_assignment(-transposed)
    pairs = []
    for uav, user in zip(rows.tolist(), cols.tolist()):
        if user < num_users:
            pairs.append((user, uav))
    return pairs

