# Solve an SDPA sparse-format (.dat-s) problem with JuMP.
#
#     julia scripts/solve_sdpa.jl <file.dat-s> [CSDP|Hypatia|SCS|Clarabel|COSMO] [solution.txt [duals.txt]]
#
# The SDPA file describes   (D)  min c.x   s.t.  Σ_j x_j F_j − F_0 ⪰ 0.
# Interior-point solvers (CSDP, Hypatia, SDPA) are fast only in their native
# form, which is the conic dual
#
#     (P)  max tr(F_0 X)   s.t.  tr(F_j X) = c_j (j = 1..m),  X ⪰ 0,
#
# with one PSD matrix variable per block and one scalar equality per SDPA
# variable.  Handing JuMP the LMI form (D) instead makes the bridges introduce
# a slack PSD variable and one equality per matrix entry, which makes memory
# grow quickly with the block size.  Strong duality (Slater holds for the
# moment problems here) makes the optimal values equal; the sign convention
# below returns the value of (D).
#
# Interior-point solvers are the right choice up to a few thousand SDPA
# variables.  For much larger problems export the moment form
# (`build_moment_form().write_json`) and use
# `scripts/solve_moment_json.py <json> 1e-5 SCS` instead.
#
# Output: every line is flushed as it is printed, and the last line is
# "done" on success.  A run whose output stops before "status ..." died inside
# the solver; one that stops after it died while writing the solution or the
# dual matrices.  The exit code is nonzero on any Julia error.

using JuMP, SparseArrays, LinearAlgebra

# print and flush at once, so that a crash inside the solver (which kills the
# process without running Julia's exit hooks) cannot swallow earlier lines
say(args...) = (println(args...); flush(stdout))

path = ARGS[1]
solver = length(ARGS) >= 2 ? uppercase(ARGS[2]) : "CSDP"

# ---------------------------------------------------------------- parse
function read_sdpa(path)
    lines = readlines(path)
    i = 1
    comment = ""
    while startswith(lstrip(lines[i]), "\"") || startswith(lstrip(lines[i]), "*")
        comment *= lines[i]
        i += 1
    end
    m = parse(Int, split(lines[i])[1]); i += 1
    _nblocks = parse(Int, split(lines[i])[1]); i += 1
    sizes = [parse(Int, t) for t in split(replace(lines[i], r"[{}(),]" => " ")) if !isempty(t)]; i += 1
    c = Float64[]
    while length(c) < m
        append!(c, parse.(Float64, [t for t in split(replace(lines[i], r"[{}(),]" => " ")) if !isempty(t)]))
        i += 1
    end
    # per block: for each matrix index (0..m) the sparse upper-triangular entries
    ent = [Dict{Int,Tuple{Vector{Int},Vector{Int},Vector{Float64}}}() for _ in sizes]
    for line in @view lines[i:end]
        p = split(line)
        length(p) < 5 && continue
        mat, blk = parse(Int, p[1]), parse(Int, p[2])
        r, cc, v = parse(Int, p[3]), parse(Int, p[4]), parse(Float64, p[5])
        e = get!(ent[blk], mat) do
            (Int[], Int[], Float64[])
        end
        push!(e[1], r); push!(e[2], cc); push!(e[3], v)
    end
    return comment, m, sizes, c, ent
end

t0 = time()
comment, m, sizes, c, ent = read_sdpa(path)
say("$path: $m variables, blocks $sizes, parsed in $(round(time() - t0, digits=1)) s")

# ---------------------------------------------------------------- model (P)
t0 = time()
model = Model()
Xs = Any[]
for (b, size) in enumerate(sizes)
    nb = abs(size)
    if size > 0
        X = @variable(model, [1:nb, 1:nb], PSD, base_name = "X$b")
    else
        X = @variable(model, [1:nb, 1:nb], base_name = "X$b")
        @constraint(model, [i in 1:nb, j in 1:nb; i != j], X[i, j] == 0)
        @constraint(model, [i in 1:nb], X[i, i] >= 0)
    end
    push!(Xs, X)
end
# tr(F X) for an upper-triangular sparse description of symmetric F
function trace_expr(F, X)
    r, cc, v = F
    e = AffExpr(0.0)
    for k in eachindex(r)
        if r[k] == cc[k]
            add_to_expression!(e, v[k], X[r[k], r[k]])
        else
            add_to_expression!(e, 2v[k], X[r[k], cc[k]])
        end
    end
    e
end
obj = AffExpr(0.0)
for b in eachindex(sizes)
    haskey(ent[b], 0) && add_to_expression!(obj, trace_expr(ent[b][0], Xs[b]))
end
@objective(model, Max, obj)
cons = Vector{Any}(undef, m)
for j in 1:m
    e = AffExpr(0.0)
    for b in eachindex(sizes)
        haskey(ent[b], j) && add_to_expression!(e, trace_expr(ent[b][j], Xs[b]))
    end
    cons[j] = @constraint(model, e == c[j])
end
say("model built in $(round(time() - t0, digits=1)) s")

# ---------------------------------------------------------------- solve
if solver == "CSDP"
    using CSDP
    set_optimizer(model, CSDP.Optimizer)
    set_attribute(model, "printlevel", 0)
elseif solver == "HYPATIA"
    using Hypatia
    set_optimizer(model, Hypatia.Optimizer)
elseif solver == "SCS"
    using SCS
    set_optimizer(model, SCS.Optimizer)
    set_attribute(model, "eps_abs", 1e-8)
    set_attribute(model, "eps_rel", 1e-8)
    set_attribute(model, "max_iters", 200000)
elseif solver == "CLARABEL"
    using Clarabel
    set_optimizer(model, Clarabel.Optimizer)
elseif solver == "COSMO"
    using COSMO
    set_optimizer(model, COSMO.Optimizer)
    set_attribute(model, "eps_abs", 1e-8)
    set_attribute(model, "eps_rel", 1e-8)
else
    error("unknown solver $solver")
end
set_silent(model)
t = @elapsed optimize!(model)
val = objective_value(model)   # = min c.x of (D) by strong duality
say("status $(termination_status(model)); optimum $val; $solver in $(round(t, digits=1)) s")
mm = match(r"value = (\S+) \+ optimum", comment)
if mm !== nothing
    relax = parse(Float64, mm.captures[1]) + val
    if occursin("negate for the maximum", comment)
        relax = -relax  # the file encodes a maximization as min of the negative
    end
    say("relaxation value $relax")
end
# the SDPA variables x are the multipliers of the equality constraints; fix
# the sign convention by matching c.x to the optimum
if length(ARGS) >= 3
    xs = [dual(cons[j]) for j in 1:m]
    if abs(sum(c .* xs) - val) > abs(sum(c .* (-xs)) - val)
        xs = -xs
    end
    open(ARGS[3], "w") do io
        for v in xs
            println(io, v)
        end
    end
    say("solution written to $(ARGS[3]) (c.x = $(sum(c .* xs)))")
end
# the PSD variables X_b are the Gram matrices of the SOS side; write them in
# the format read by ncalg::ncsos::read_dual_blocks
if length(ARGS) >= 4
    open(ARGS[4], "w") do io
        for (b, X) in enumerate(Xs)
            V = value.(X)
            n = size(V, 1)
            println(io, "block $(b - 1) $n real")
            for r in 1:n
                println(io, join((string(V[r, cc]) for cc in 1:n), " "))
            end
        end
    end
    say("dual matrices written to $(ARGS[4])")
end
say("done")
