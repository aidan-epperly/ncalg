#!/usr/bin/env python
"""Solve an SDPA sparse-format (.dat-s) problem with CVXPY.

    python scripts/solve_sdpa.py <file.dat-s> [SCS|CLARABEL] [eps] [solution.txt [duals.txt]]

The SDPA problem is  minimize c.x  subject to  X = sum_i x_i F_i - F_0 >= 0
(one PSD block per block in the file; negative block sizes are diagonal).
Prints the optimum and the relaxation value encoded in the comment line.
"""
import re
import sys
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def read_sdpa(path):
    comment = ""
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while lines[i].lstrip().startswith(('"', "*")):
        comment += lines[i].strip().strip('"*') + " "
        i += 1
    m = int(lines[i].split()[0]); i += 1
    nblocks = int(lines[i].split()[0]); i += 1
    sizes = [int(t) for t in re.split(r"[\s,{}()]+", lines[i].strip()) if t]; i += 1
    c = []
    while len(c) < m:
        c += [float(t) for t in re.split(r"[\s,{}()]+", lines[i].strip()) if t]
        i += 1
    c = np.array(c)
    # entries[mat][block] -> (rows, cols, vals)
    entries = {}
    for line in lines[i:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        mat, blk, r, col, val = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
        entries.setdefault((mat, blk), ([], [], []))
        rows, cols, vals = entries[(mat, blk)]
        rows.append(r - 1); cols.append(col - 1); vals.append(val)
    return comment, m, sizes, c, entries


def block_matrix(entries, mat, blk, size):
    rows, cols, vals = entries.get((mat, blk), ([], [], []))
    a = sp.coo_matrix((vals, (rows, cols)), shape=(size, size))
    # entries give the upper triangle; symmetrize without doubling the diagonal
    d = sp.diags(a.diagonal())
    return (a + a.T - d).tocsr()


def main():
    path = sys.argv[1]
    solver = (sys.argv[2] if len(sys.argv) > 2 else "SCS").upper()
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    comment, m, sizes, c, entries = read_sdpa(path)
    print(f"{path}: {m} variables, blocks {sizes}")
    x = cp.Variable(m)
    constraints = []
    for b, size in enumerate(sizes, start=1):
        n = abs(size)
        f0 = block_matrix(entries, 0, b, n)
        # X = sum_i x_i F_i - F_0 ; build as a sparse linear map on vec(X)
        cols = []
        for i in range(1, m + 1):
            fi = block_matrix(entries, i, b, n)
            cols.append(fi.reshape((n * n, 1)))
        a = sp.hstack(cols).tocsc()
        xvec = a @ x - f0.reshape((n * n, 1)).toarray().ravel()
        xmat = cp.reshape(xvec, (n, n), order="F")
        if size > 0:
            constraints.append(xmat >> 0)
        else:
            constraints.append(cp.diag(xmat) >= 0)
    prob = cp.Problem(cp.Minimize(c @ x), constraints)
    t = time.time()
    if solver == "SCS":
        prob.solve(solver=cp.SCS, eps=eps, max_iters=200000, verbose=False)
    elif solver == "CLARABEL":
        prob.solve(solver=cp.CLARABEL, verbose=False)
    else:
        prob.solve(solver=solver, verbose=False)
    print(f"status {prob.status}, optimum {prob.value:.12f}, {solver} in {time.time() - t:.1f} s")
    mo = re.search(r"value = (\S+) \+ optimum", comment)
    if mo:
        val = float(mo.group(1)) + prob.value
        if "negate for the maximum" in comment:
            val = -val  # the file encodes a maximization as min of the negative
        print(f"relaxation value {val:.12f}")
    if len(sys.argv) > 4:
        with open(sys.argv[4], "w") as f:
            f.write("\n".join(f"{v:.17e}" for v in np.asarray(x.value).ravel()) + "\n")
        print(f"solution written to {sys.argv[4]}")
    if len(sys.argv) > 5:
        # dual matrices (Gram matrices of the SOS side), one per block, in the
        # format read by ncalg::ncsos::read_dual_blocks
        with open(sys.argv[5], "w") as f:
            for b, con in enumerate(constraints):
                Y = np.atleast_2d(np.asarray(con.dual_value))
                if Y.shape[0] != Y.shape[1]:
                    Y = np.diag(Y.ravel())
                n = Y.shape[0]
                f.write(f"block {b} {n} real\n")
                for r in range(n):
                    f.write(" ".join(f"{Y[r, c]:.17e}" for c in range(n)) + "\n")
        print(f"dual matrices written to {sys.argv[5]}")


if __name__ == "__main__":
    main()
