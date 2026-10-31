#!/usr/bin/env python
"""Solve an ncalg SDP exported as JSON (`MomentSdp::write_json`,
`MomentForm::write_json` or `SdpModel::write_json`) in CVXPY: one PSD
constraint per block, real blocks as real symmetric expressions, complex
Hermitian blocks as their real embedding `[[Re, -Im], [Im, Re]]` (built here,
not by CVXPY), plus optional linear equalities.

With `--duals` the dual matrix of every block is written in the format of
`ncalg::ncsos::read_dual_blocks`: `n x n` for real blocks, the real
`2n x 2n` dual of the embedding for complex ones, which
`MomentSdp::dual_grams` unembeds (`unembed_dual`).  Building the embedding
here keeps the duals in ncalg's `tr(G F^T)` convention (CVXPY's own complex
PSD duals pair as `Re tr(F Y)`).

    python scripts/solve_moment_json.py <file.json> [eps] [SCS|CLARABEL|...] [--out solution.txt]

Prints compile time, solver time, iterations and the objective value; with
`--out` the variable vector is written one number per line (readable with
`ncalg::ncsos::read_solution`).
"""
import json
import sys
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def load(path):
    with open(path) as f:
        return json.load(f)


def build(data):
    k = data["num_vars"]
    z = cp.Variable(k)
    constraints = []
    for blk in data["blocks"]:
        n = blk["size"]
        real = blk["real"]
        dtype = float if real else complex
        rows, cols, vals = [], [], []
        f0 = np.zeros(n * n, dtype=dtype)
        for r, c, re, im in blk["f0"]:
            v = re if real else complex(re, im)
            f0[r * n + c] += v
            if r != c:
                f0[c * n + r] += np.conj(v) if not real else v
        for j, entries in enumerate(blk["fj"]):
            for r, c, re, im in entries:
                v = re if real else complex(re, im)
                rows.append(r * n + c); cols.append(j); vals.append(v)
                if r != c:
                    rows.append(c * n + r); cols.append(j); vals.append(np.conj(v) if not real else v)
        T = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, k), dtype=dtype)
        if real:
            if n == 1:
                constraints.append(T @ z + f0 >= 0)
                continue
            M = cp.reshape(T @ z + f0, (n, n), order="C")
        else:
            # real embedding [[A, -B], [B, A]] of A + iB, as one real 2n x 2n expression
            Tr, Ti = sp.csr_matrix(T.real), sp.csr_matrix(T.imag)
            A = cp.reshape(Tr @ z + f0.real, (n, n), order="C")
            B = cp.reshape(Ti @ z + f0.imag, (n, n), order="C")
            M = cp.bmat([[A, -B], [B, A]])
        constraints.append(M >> 0)
    # optional linear equalities E z = e
    eqs = data.get("equalities", [])
    if eqs:
        rows, cols, vals, rhs = [], [], [], []
        for q, eq in enumerate(eqs):
            for i, cval in eq["coeffs"]:
                rows.append(q); cols.append(i); vals.append(cval)
            rhs.append(eq["rhs"])
        E = sp.csr_matrix((vals, (rows, cols)), shape=(len(eqs), k))
        constraints.append(E @ z == np.array(rhs))
    c = np.array(data["objective"])
    prob = cp.Problem(cp.Minimize(c @ z), constraints)
    return prob, z


def write_duals(path, mats):
    """Dual matrices, one block per PSD constraint, in the format read by
    `ncalg::ncsos::read_dual_blocks`.

    All blocks are real here (complex ones are embedded), so each dual is a
    real symmetric matrix; `dual_grams` unembeds the `2n x 2n` ones."""
    with open(path, "w") as f:
        for b, Y in enumerate(mats):
            Y = np.atleast_2d(np.asarray(Y))
            if Y.shape[0] != Y.shape[1]:
                Y = np.diag(Y.ravel())
            Y = Y.T  # tr(G F^T) convention (real here: only symmetrizes the layout)
            n = Y.shape[0]
            cplx = np.iscomplexobj(Y)
            f.write(f"block {b} {n} {'complex' if cplx else 'real'}\n")
            for r in range(n):
                if cplx:
                    f.write(" ".join(f"{Y[r, c].real:.17e} {Y[r, c].imag:.17e}" for c in range(n)) + "\n")
                else:
                    f.write(" ".join(f"{Y[r, c]:.17e}" for c in range(n)) + "\n")


def main():
    args = [a for a in sys.argv[1:]]
    out = None
    duals = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    if "--duals" in args:
        i = args.index("--duals")
        duals = args[i + 1]
        del args[i:i + 2]
    path = args[0]
    eps = float(args[1]) if len(args) > 1 else 1e-6
    solver = (args[2] if len(args) > 2 else "SCS").upper()
    data = load(path)
    t = time.perf_counter()
    prob, z = build(data)
    t_build = time.perf_counter() - t
    if solver == "SCS":
        prob.solve(solver=cp.SCS, eps=eps, max_iters=200000)
    else:
        prob.solve(solver=solver)
    st = prob.solver_stats
    value = data["objective_constant"] + prob.value
    if not data["minimize"]:
        value = -value
    print(
        f"{path}: {data['num_vars']} vars, blocks {[b['size'] for b in data['blocks']]}, "
        f"{len(data.get('equalities', []))} equalities; "
        f"model {t_build:.2f} s, compile {prob.compilation_time:.2f} s, {solver} solve {st.solve_time:.3f} s, "
        f"iters {st.num_iters}, status {prob.status}"
    )
    print(f"objective value {value:.12f}")
    if out is not None:
        with open(out, "w") as f:
            for v in np.asarray(z.value).ravel():
                f.write(f"{v:.17e}\n")
        print(f"solution written to {out}")
    if duals is not None:
        nb = len(data["blocks"])
        write_duals(duals, [prob.constraints[b].dual_value for b in range(nb)])
        print(f"dual matrices written to {duals}")


if __name__ == "__main__":
    main()
