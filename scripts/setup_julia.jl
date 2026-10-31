# One-time setup of the Julia SDP / NC-SOS stack:  julia scripts/setup_julia.jl
using Pkg
Pkg.add(["JuMP", "CSDP", "SDPA", "Hypatia", "SCS", "Clarabel", "COSMO"])
Pkg.status()
