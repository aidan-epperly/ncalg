# ncalg — noncommutative algebra in safe Rust

Polynomials in free associative algebras `K<x_1, …, x_n>` with integer,
rational, floating point and complex coefficients; noncommutative Gröbner
bases (Mora's algorithm) that can be bounded and resumed so an infinite basis
never means an infinite runtime; and a lazy evaluator that **reduces modulo a
Gröbner basis while an expression is being expanded**, so the unreduced
expansion is never built.

The crate is `#![forbid(unsafe_code)]`.

## Contents

- [Quick start](#quick-start)
- [Coefficients](#coefficients)
- [Polynomials](#polynomials)
- [Commutative polynomials](#commutative-polynomials-ncalgcommutative)
- [Monomial orderings](#monomial-orderings)
- [Gröbner bases](#gröbner-bases)
  - [Partial bases and limits](#partial-bases-and-limits)
  - [What a truncated basis guarantees](#what-a-truncated-basis-guarantees)
  - [The algorithm](#the-algorithm)
- [Reduction while expanding](#reduction-while-expanding)
- [Quotient algebras, normal words, Hilbert series](#quotient-algebras-normal-words-hilbert-series)
- [Saving and loading](#saving-and-loading)
- [NC-SOS / moment relaxations](#nc-sos--moment-relaxations-ncalgncsos)
- [General SDP models](#general-sdp-models-ncalgsdp)
  - [SOS certificates](#sos-certificates)
- [Native SDP solver](#native-sdp-solver-ncalgsolver)
- [GNS state extraction](#gns-state-extraction-ncalggns)
- [Linear algebra](#linear-algebra-ncalglinalg)
- [Parallelism](#parallelism)
- [Exact arithmetic over finite rings](#exact-arithmetic-over-finite-rings-ncalgarith)
- [Finite fields](#finite-fields-ncalgff)
- [Groups and symmetry](#groups-and-symmetry-ncalgsymmetry)
- [Polycyclic groups](#polycyclic-groups-ncalgpcgroup)
- [Linear algebra over ℤ/N and ℤ](#linear-algebra-over-ℤn-and-ℤ-ncalgzmod)
- [Finite abelian groups](#finite-abelian-groups-ncalgabelian)
- [Cyclotomic numbers](#cyclotomic-numbers-ncalgcyclo)
- [Built-in relation sets](#built-in-relation-sets)
- [Examples and benchmarks](#examples-and-benchmarks)
- [Testing](#testing)
- [Bugs and improvements](#bugs-and-improvements)
- [Design notes and limitations](#design-notes-and-limitations)
- [Elimination of moment constraints](#elimination-of-moment-constraints)
- [License](#license)

## Quick start

```rust
use ncalg::{FreeAlgebra, Polynomial, GroebnerBasis, GroebnerOptions, BigRational};

// Q<x, y> with the deglex ordering x < y
let alg = FreeAlgebra::new(["x", "y"]);
let x: Polynomial<i64> = alg.gen("x");
let y: Polynomial<i64> = alg.gen("y");

// arithmetic is noncommutative
let c = &x * &y - &y * &x;
assert_eq!(c, alg.parse("[x, y]").unwrap());

// Gröbner basis of the Weyl algebra  y x - x y = 1  (over Q)
let rel: Polynomial<BigRational> = alg.parse("y*x - x*y - 1").unwrap();
let gb = GroebnerBasis::compute(vec![rel], &GroebnerOptions::default());
assert!(gb.is_complete());

// normal form of y^2 x^2, reduced while the product is expanded
let f: Polynomial<BigRational> = alg.parse("y^2").unwrap();
let g: Polynomial<BigRational> = alg.parse("x^2").unwrap();
assert_eq!(gb.reduce_product(&f, &g).to_string(), "x^2*y^2 + 4*x*y + 2");
```

Build and test with the usual commands (`cargo build`, `cargo test`,
`cargo doc --open`).  Examples live in `examples/`.

## Coefficients

Coefficient types implement the `Coeff` trait (a commutative ring); those that
also implement `Field` can be used for Gröbner bases and reduction.

| type | ring | field | notes |
|---|---|---|---|
| `i64`, `i128` | ✓ | | checked arithmetic: overflow panics rather than wrapping |
| `BigInt` | ✓ | | arbitrary precision |
| `BigRational` | ✓ | ✓ | exact; what integer polynomials are promoted to for Gröbner bases |
| `HybridRational` | ✓ | ✓ | exact rationals in `i64` words with `i128` intermediates, promoted to `BigRational` only on overflow (unique representation, structural equality); 1.1–2.3× faster Gröbner bases than `BigRational` (`examples/coeff_bench.rs`) |
| `f64` | ✓ | ✓ | relative (scale-invariant) cancellation test; optional absolute floor `set_float_tolerance` (default `0`) |
| `Complex<f64>` | ✓ | ✓ | |
| `Complex<BigRational>`, `Complex<HybridRational>` | ✓ | ✓ | exact Gaussian rationals, e.g. for Pauli relations |
| `Fp<P>` | ✓ | ✓ | prime field `Z/pZ` (`P < 2^63`, alias of `ff::Fq<P, 1>`), fast exact arithmetic for exploratory computations |
| `ff::Fq<P, K>` | ✓ | ✓ | GF(p^k) (table arithmetic for q ≤ 2^22), printed and parsed as `#e`, see [Finite fields](#finite-fields-ncalgff) |

Conversions: `Polynomial<i64>::to_rational()`, `to_f64()`, generic
`convert::<D>()`, and `Polynomial<BigRational>::to_integer_primitive()` to go
back to primitive integer polynomials.  `GroebnerBasis::compute_from_integers`
runs the whole pipeline for integer input, and `integer_elements()` returns the
basis scaled to integers.

The numeric types are re-exported (`ncalg::{BigInt, BigRational, HybridRational, Complex}`).
All exact types convert into each other with `convert::<D>()` (`HybridRational`
is the recommended default for exact computations; switch to `BigRational`
only if a profile shows promotion happening).

## Polynomials

`FreeAlgebra` is a cheap handle holding generator names and the monomial
ordering.  `Polynomial<C>` stores terms sorted in strictly decreasing monomial
order with no zero coefficients, so the leading term is `terms()[0]` and
structural equality is polynomial equality.

Operations (all available through `std::ops` where sensible):

- `+ - *` between polynomials (owned or borrowed), `-p`, `+= -= *=`;
- scalar multiplication `2 * &p`, `&p * c`, `p + c`;
- `pow`, `commutator`, `anticommutator`, `mul_word_left/right`, `sandwich(l, r)`;
- `reversed()`, `adjoint()` (reverse words, conjugate coefficients), `is_hermitian()`;
- `substitute(var, poly)`, `substitute_with(closure)`, `derivation(var)`
  (the free derivation with `∂x_v = 1`);
- `homogeneous_components()`, `truncate_degree(d)`, `map_coeffs`, `convert`;
- `leading_term/word/coeff`, `tail`, `coeff_of(word)`, `degree`, `support_vars`;
- `left_quotient_by_word` / `right_quotient_by_word` (exact division by a word);
- `evaluate(image, embed, zero, one)` in any ring, e.g. matrices;
- `to_latex()`, `Display` (`x^2*y - 1/2*y*x + 3`), `Hash`/`Eq` for hashable coefficients;
- `content()` / `primitive_part()` for integer coefficients, `monic()` for fields.

Parsing: `alg.parse::<C>("2*x*y - y*x + x^2 - 1/2")`.  Juxtaposition
multiplies (`x y`), `^` takes a non-negative integer exponent, `[a, b]` is the
commutator, `{a, b}` the anticommutator, `i` the imaginary unit for complex
coefficient types, and a postfix `'` is the adjoint (`(a*n)'`).

**Involutions.** By default every generator is Hermitian. `FreeAlgebra::new_star`
takes `(name, is_hermitian)` pairs and creates a partner `a†` for each
non-Hermitian `a` with `a* = a†`; `with_order_and_adjoint` gives full control.
`Polynomial::adjoint`, `is_hermitian`, the relaxation builder, `is_star_closed`
and the parser's `'` all use the algebra's involution.

## Commutative polynomials (`ncalg::commutative`)

For variables that commute anyway, two direct types (no quotient algebra
needed), over any `Coeff`:

- `Laurent<C>`: `Σ c_k t^k` with `k ∈ ℤ`, stored densely, so equality is
  structural.  `+ - *`, `pow`, `shift`, `substitute_power` (`t ↦ t^k`),
  `negate_variable`, `div_exact`, evaluation in a field (`eval_with`), and
  `display_as(var, den)`, which prints exponents as fractions `k/den`.
- `MPoly<C>`: polynomials in `x_0, x_1, …`.  `+ - *` (also with `i64`),
  `pow`, `degree`, `terms`, `eval_with` in any ring, and `display(&names)`.
  Over a field, `MPoly::parse("n(n+1)/2 - 2m", &names)` reads numbers,
  names, `+ - * /` (division by constants), `^`, parentheses and implicit
  products; errors carry a position like the free-algebra parser's.  For
  `MPoly<BigRational>`, `eval_i64`, `eval_integer` and `parity` handle
  integer-valued polynomials in integer parameters.

```rust
use ncalg::commutative::MPoly;
use ncalg::BigRational;

let names = ["n".to_string(), "m".to_string()];
let w = MPoly::<BigRational>::parse("2nm + n + m", &names).unwrap();
assert_eq!(w.eval_integer(&[2, 3]), Some(17.into()));
assert_eq!(w.parity(&[None, Some(true)]), None);       // n + 1 when m is odd
```

## Monomial orderings

All orderings are admissible (well-founded total orders compatible with
multiplication on both sides) and are used through the object-safe
`MonomialOrder` trait.

| ordering | description | degree compatible |
|---|---|---|
| `DegLex` (default) | length, then left-to-right lexicographic, `x_0 < x_1 < …` or a custom chain | yes |
| `DegRevLex` | length, then right-to-left with the smaller generator winning | yes |
| `WeightedDegLex` | weighted degree, then length, then lex | only for equal weights |
| `Elimination(k)` | number of letters from the first `k` generators, then deglex | no |
| `Wreath` | wreath-product ordering (count of the largest letter, then segments recursively); orients power-commutator relations | no |

`FreeAlgebra::with_order(names, order)` selects one.  Under `Elimination(k)`
the basis elements free of the first `k` generators generate the elimination
ideal.

## Gröbner bases

```rust
let gb = GroebnerBasis::compute(generators, &GroebnerOptions::default());
gb.is_complete();           // true if every obstruction reduced to zero
gb.elements();              // monic, interreduced, sorted by leading word
gb.reduce(&p);              // normal form
gb.contains(&p);            // ideal membership (exact when complete)
gb.verify(None);            // brute-force re-check of all obstructions
gb.is_star_closed();        // is the ideal closed under the Hermitian adjoint?
gb.convert::<f64>();        // same basis with other coefficients
```

For exploratory work where rational coefficient growth is the bottleneck,
compute over a prime field: `Polynomial<Fp<32003>>` behaves like any other
coefficient type and the leading words / Hilbert series agree with the
rational answer for all but finitely many primes.

### Partial bases and limits

Noncommutative Gröbner bases may be infinite, so every computation can be
bounded.  Options are set with a builder:

```rust
let opts = GroebnerOptions::new()
    .max_degree(8)                        // skip obstructions with overlap word longer than 8
    .max_pairs(10_000)                    // at most this many S-polynomial reductions per run
    .max_basis_size(500)
    .time_limit(Duration::from_secs(30))
    .cancel_token(flag.clone());          // Arc<AtomicBool> set from another thread
```

The result always comes back, with `termination()` telling you why the
computation stopped (`Complete`, `DegreeBound { bound, smallest_pending }`,
`PairLimit`, `BasisSizeLimit`, `TimeLimit`, `Canceled`) and `stats()`
describing the work done.  A `GroebnerSolver` keeps the pending obstruction
queue so the computation can be **resumed** after relaxing the limits, and
new generators can be added with `add_generators`:

```rust
let mut solver = GroebnerSolver::new(gens, GroebnerOptions::new().max_degree(6));
solver.run();                       // DegreeBound
solver.set_options(GroebnerOptions::new().max_degree(12));
solver.run();                       // continues from the pending queue
let gb = solver.basis();
```

### What a truncated basis guarantees

Obstructions are processed in order of increasing overlap-word length, and
`max_degree = d` stops as soon as every pending obstruction has an overlap word
longer than `d`.  The returned set `G_d` therefore has every obstruction of
overlap degree ≤ d reducing to zero.

- For a degree-compatible ordering and **homogeneous** generators, `G_d` is a
  Gröbner basis of the ideal "up to degree d": for every polynomial `p` of
  degree ≤ d, `p ∈ I` iff `reduce(p) = 0`, normal forms of such `p` are unique,
  and the normal words of length ≤ d are a basis of the quotient in those
  degrees.  `G_d` consists exactly of the elements of the reduced Gröbner basis
  that have degree ≤ d.
- For inhomogeneous generators `G_d` is still a subset of the reduced basis and
  gives valid (but not necessarily canonical) coset representatives;
  `contains` can return false negatives.

`GroebnerBasis::verify(Some(d))` independently re-checks the property by brute
force.

### The algorithm

Mora's completion procedure for free algebras (the noncommutative Buchberger
algorithm):

1. Generators are reduced and inserted one at a time.  The active basis is
   kept **interreduced**: inserting `h` removes every element whose leading
   word contains `lw(h)`; the removed elements are re-reduced and re-inserted
   if nonzero, so inclusion obstructions never need to be considered.
2. For each pair of active elements (including an element with itself), every
   proper overlap `lw(g_i) = a·c`, `lw(g_j) = c·b` yields the obstruction
   `S = g_i·b − a·g_j` with overlap word `w = a·c·b`.  Obstructions sit in a
   priority queue keyed by `|w|` (then FIFO).
3. A noncommutative **chain criterion** skips an obstruction when the leading
   word of some active element occurs strictly inside `w` (not at the two
   defining positions): its S-polynomial is then `S_ik·b' + a'·S_kj` for two
   obstructions with shorter overlap words, so it is redundant.  It can be
   turned off with `.chain_criterion(false)`; the test suite checks that both
   settings give the same (unique) reduced basis.
4. S-polynomials are formed *inside the lazy reducer*: the two canceling
   leading terms are never materialized, only the tails are streamed.
5. On completion the basis is tail-reduced so the output is the reduced
   Gröbner basis (unique for the ideal and ordering).

Leading-word lookups use Aho–Corasick automata over the active leading
words, so finding a divisor of a word costs `O(|word|)` per automaton
regardless of basis size.  While the basis grows, the solver keeps them by the
Bentley–Saxe logarithmic method (`automaton::DynamicMatcher`, `Reducer::new_dynamic`):
static automata on groups of at most `2^i` words, merged like binary carries on
insertion, with retired words skipped and a group rebuilt once half of it is
retired.  An insertion costs `O(L log N)` amortized instead of a rebuild, a
lookup `O(|word| log N)` (32-qubit Pauli basis: 0.74 s instead of 10.3 s with rebuilds).

## Reduction while expanding

Every operation in the library is expressed as a sum of *scaled sandwiches*
`c · a · p · b` (`c` a coefficient, `a`, `b` words, `p` a sorted polynomial).
Because the ordering is admissible, `a · p · b` has its terms in the same order
as `p`, so each sandwich is a sorted stream.  An `Accumulator` is a max-heap of
such streams; pulling from it yields the terms of the entire sum in decreasing
order, merging equal words as they meet at the top (Monagan–Pearce heap
multiplication when the streams are `f_i · g`).

A `Reducer` consumes an accumulator and watches each term `c·w` that emerges.
If `w = a · lw(g) · b` for a basis element `g` (found by the automaton), it
pushes the stream `−c/lc(g) · a · tail(g) · b` back into the same heap: the
leading terms cancel exactly and are never materialized.  Otherwise the term
is appended to the remainder, which therefore comes out sorted.  The
unreduced product never exists in memory; the peak heap size is bounded by the
number of live streams, not by the size of the free expansion.

Entry points:

| call | computes |
|---|---|
| `reducer.reduce(&p)` | normal form of `p` |
| `reducer.reduce_product(&f, &g)` | normal form of `f·g` without forming `f·g` |
| `reducer.reduce_pow(&f, n)` | binary powering, reducing after every squaring |
| `reducer.reduce_commutator / reduce_anticommutator` | both products streamed into one heap |
| `reducer.reduce_product_chain(iter)` | `f_1 ⋯ f_k`, reducing after every factor |
| `reducer.run(accumulator, mode)` | anything you can express as streams; `ReductionMode::Top` stops at an irreducible leading term |
| `Expr::eval_mod(&reducer)` | an expression tree (sums, products, powers, commutators, scalings, adjoints) |
| `QuotientAlgebra` / `QuotientElement` | ring arithmetic where every element is always in normal form |

`Expr` flattens sums of products into one heap pass, folds product chains
with reduction after each factor, and pushes adjoints down to the leaves
(since the adjoint of a normal form is not the normal form of the adjoint
unless the ideal is `*`-closed).

On the 2-qubit Pauli algebra, `H^10` for a five-term Hamiltonian takes under a
millisecond lazily versus ~20 s for expanding (9.7 million terms) and then
reducing; see `examples/bench.rs`.

## Quotient algebras, normal words, Hilbert series

```rust
let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
let x = q.gen("X0");
let y = q.gen("Y0");
let z = (&x * &y).pow(3) - x.commutator(&y);   // always in normal form
q.normal_words(3);        // basis words of length 3
q.hilbert_series(10);     // number of normal words of each length 0..=10
q.dimension();            // Some(n) if finite dimensional, None otherwise

// NPA / NC-SOS style moment matrices: normal forms of u† v over an index set
let index = q.normal_words_up_to(2);
let entries = q.adjoint_products(&index);          // entries[i][j] = NF(index[i]† index[j])
let row = entries[1][2].coefficient_vector(&q.normal_words_up_to(4)); // dense vector over a word basis
```

Normal words and their counts are computed on the leading-word automaton:
counts by dynamic programming over states (no enumeration), finiteness by
cycle detection on the graph of non-terminal states.

## Saving and loading

Polynomials and Gröbner bases (complete or partial) can be written to a
plain-text `ncalg v1` file and read back, including the algebra definition
(generator names, monomial ordering, involution) and the termination status:

```rust
gb.save("sl2.ncalg")?;                                       // GroebnerBasis<C>
let gb = GroebnerBasis::<BigRational>::load("sl2.ncalg")?;   // same coefficient type required
let gbf = GroebnerBasis::<f64>::load_as("sl2.ncalg")?;       // convert while loading
ncalg::io::save_polynomials("rels.ncalg", &alg, &rels)?;
let (alg, rels) = ncalg::io::load_polynomials::<BigRational>("rels.ncalg")?;
```

The file is a short header followed by one polynomial per line in the syntax
the parser accepts, so it is human-readable and editable:

```
ncalg v1
kind: groebner
generators: e, f, h
order: deglex
adjoint: 0 1 2
coefficients: BigRational
termination: complete
count: 3
f*e - e*f + h
h*e - e*h - 2*e
h*f - f*h + 2*f
```

Orderings are written as `deglex`, `degrevlex`, `deglex(rank=[…])`,
`wdeglex([…])` or `elim(k)` (see `ncalg::order::parse_order`); a partial
basis records `termination: degree-bound <bound> <smallest pending degree>`
or `pair-limit`, `basis-size-limit`, `time-limit`, `canceled`.

## NC-SOS / moment relaxations (`ncalg::ncsos`)

Given any `Reducer` (a complete Gröbner basis, or a partial one, which still
yields a valid but weaker relaxation) and an index set of words, the module
builds the Lasserre/NPA moment relaxation

```text
min L(h)  s.t.  L(1) = 1,  L(p*) = conj L(p),  [L(u* v)]_{u,v∈W} ⪰ 0,  [L(u* g_k v)] ⪰ 0
```

with every product reduced to normal form, the Hermitian-symmetry equalities
eliminated exactly, complex coefficients handled by the real embedding, and
`L(1) = 1` folded into the constant matrix. The result is an LMI
`F_0 + Σ z_j F_j ⪰ 0` exported in SDPA sparse format, whose conic dual is the
SOS / Gram form.

```rust
use ncalg::ncsos::Relaxation;
let sdp = Relaxation::level(gb.reducer(), 2)       // index set: normal words of degree <= 2
    .localizing_level(&g, 1)                         // optional constraint g >= 0
    .minimize(&h)
    .build();
sdp.write_sdpa("problem.dat-s")?;
// after solving: bound = sdp.relaxation_value(optimum); moments: sdp.moments(&z)
```

**Symmetry reduction.** `Relaxation::symmetry(generators)` takes generators of
a finite group of permutations of the algebra's generators that preserve the
ideal, the objective and the localizing polynomials (all verified). The
moment variables are then averaged over the group exactly (orbit reduction,
computed one orbit at a time), and each matrix is block-diagonalized along the
isotypic decomposition of the group's action on the index words, keeping one
copy of each irreducible block (disable with `.block_diagonalize(false)`).
When the group acts on the index words by signed permutations the
decomposition is exact and orbit by orbit: each orbit is a quotient of the
regular representation, and its basis vectors come from the group's
irreducible representations, which keeps the build fast even for relaxations
with hundreds of thousands of variables.
Other actions use a numerical decomposition. On the
8-cycle Quantum Max Cut relaxation the dihedral group of order 16 cuts the
problem from 749 variables and one 295×295 block to 79 variables and seven
blocks of size at most 78, with the same optimum. Signed permutations
(`x_v ↦ ±x_{π(v)}`, via `Relaxation::signed_symmetry` and
`ncsos::SignedPermutation`) cover parities such as the Majorana `γ_a ↦ −γ_a`,
which reproduces the even/odd parity-block split. Helpers:
`ncsos::permutation_group`, `ncsos::signed_permutation_group` and
`ncsos::pair_permutation` (for `s_ij`-type generators).
When the group acts on the index words by signed permutations, each reduced
block is `V^† M V` with an orthonormal `V` (`MomentSdp::block_vectors`), so
`M` is unitarily equivalent to `⊕_b I_{d_b} ⊗ M_b` (`d_b` =
`BlockInfo::irrep_dimension`) and constraints added on the reduced blocks such
as `I − M_b ⪰ 0` or `Σ_b d_b tr Y_b = 1` mean the same as on `M`. If a
permuted word reduces to a combination of several words, the blocks are only
congruent to the restriction: `M_b ⪰ 0` is still exact, but such added
constraints are not.

**Connected components.** Independently of any symmetry, `build()` splits
every matrix into the connected components of its sparsity pattern: index
words `u`, `v` are linked when the entry `L(u* g v)` is not identically zero
after the exact elimination. This is exact and catches block structure that is
not a declared group, e.g. index words `P_a w` for central orthogonal
idempotents `P_a` (several functionals encoded in one), which give one block
per `a`. With a symmetry, components that the group permutes are
block-diagonalized together. Disable with `.split_components(false)`.

Moment constraints (`moment_equality`) must each be invariant under the group.
When they come in orbits instead (one constraint per site, say), add them all
and call `.symmetrize_constraints(true)`: every constraint is replaced by its
group average, zeros are dropped and scalar multiples deduplicated. This is
equivalent for the invariant functionals the relaxation works with, provided
the constraint set is invariant as a set, which is checked
(`effective_constraints()` lists the result, `group_elements()` the group).

**Several functionals.** `ncalg::multi::MultiRelaxation` holds one ordinary
`Relaxation` per pseudo-state (own index set, localizing matrices, moment
constraints, symmetry and `L_k(1) = 1`), builds each with `try_build` and
stacks them in one `SdpModel` with links `Σ_k L_k(p_k) = v` and an objective
`Σ_k L_k(h_k)`: the layout of Markovian / differential-NPO hierarchies,
sparse NPA and steady-state bounds, without encoding the functionals by
central idempotents.

```rust
let mut mr = MultiRelaxation::new(gb.reducer());
let a = mr.functional("a", Relaxation::level(gb.reducer(), 1));
let b = mr.functional("b", Relaxation::level(gb.reducer(), 1));
mr.link(&[(a, &za), (b, &zb)], C::new(0.0, 0.0));
mr.maximize(&[(a, &h)]);
let sdp = mr.build()?;
let (value, sol) = sdp.solve(&SolverOptions::default())?;
let cert = sdp.certificate(&sol.duals, gb.reducer(), 1e-9);   // one SOS per functional
cert.bound; cert.residual_l1;                  // + link multipliers, constants
```

`MultiSdp::certificate` turns the dual matrices (native, or read from CSDP)
into one SOS certificate per functional plus a constant per functional and a
multiplier per link and per own constraint, which the elimination removed and
which are recovered by least squares on the identity
`h_k − λ_k − Σ (conj(μ) p + μ p*)/2 = ±σ_k`.  It reports the bound and the
residual; `residual_l1` bounds the residual's effect when the moments of its
words are bounded by 1 (Pauli strings, unitaries).  Differential-NPO
relaxations have no strictly feasible point, so solve them with
`regularization: 1e-8`; the Gram matrices of the plain solve diverge and give
a poor certificate.  `examples/multi_quench.rs` is a complete small case: a
one-qubit quench in the differential-NPO hierarchy of Araújo, Garner and
Navascués (arXiv:2408.02572), bracketed from both sides and certified.

**Two output forms.** `build()` eliminates the Hermitian/symmetry equalities
exactly and exports an LMI (`write_sdpa`, `write_json`). `build_moment_form()`
instead keeps the moment values themselves as variables (real and imaginary
parts) with `L(1) = 1`, Hermiticity and symmetry as explicit equality
constraints, exported with `MomentForm::write_json`; this is the layout most
hand-written NPA codes use and is read by `scripts/solve_moment_json.py`.

Solver scripts for `.dat-s` files are in `scripts/`: `solve_sdpa.py`
(CVXPY with SCS/Clarabel; writes the solution vector) and `solve_sdpa.jl`
(JuMP in the solver-native Gram form for CSDP, Hypatia, SCS, Clarabel, COSMO).
`julia scripts/setup_julia.jl` installs the Julia packages once.

## Linear algebra (`ncalg::linalg`)

`linalg` holds the linear algebra shared across ncalg. Each algorithm has one implementation:

- **`linalg::sparse`:**
  - the storage types `SparseHerm` and `SparseSym` (also at `sdp::` and `ncsos::`);
  - `full_entries`, `to_dense` and `norm_sq` for a stored upper triangle;
  - `SparseEchelon`, incremental sparse Gaussian elimination with row scaling and partial pivoting;
  - `rref_sparse`, built on `SparseEchelon`.

  `SparseEchelon` is the one sparse elimination. It is behind `SdpModel::eliminate`, the kernel basis of the eliminated moment relaxation, and the choice of independent Gram multipliers.
- **`linalg::sparse_qr`** is the rank-revealing sparse QR (also at `ncalg::sparse_qr`), for large redundant floating-point systems (`SdpModel::eliminate_qr`, the moment constraints).
- **Dense products, Hermitian eigendecompositions and least squares** live in the crate-internal `linalg::dense`. Least squares goes through QR (TSQR, then a truncated SVD of `R`), never the normal equations.
- **`linalg::exact`:** dense `rref`, `rank`, `nullspace`, `solve`, `inverse`, `det` and `mul` over any exact `Field` (rationals, `Cyclo<N>`, `Fq<P, K>`), with first-nonzero pivoting and Bareiss fraction-free elimination for the determinant, so intermediate entries are minors. It refuses floating-point types: that pivoting is not backward stable, and dense floating point belongs to faer. `ff::FqMat` stays separate for fields chosen at run time.

The SDP stack lives under `sdp`: the model (`ncalg::sdp`), the native solver (`sdp::solver`, also at `ncalg::solver`) and its facial reduction.

## General SDP models (`ncalg::sdp`)

The moment relaxation is one instance of a small modeling layer: real
variables, Hermitian or real symmetric matrix variables, LMIs on affine
matrix expressions, linear equalities and a linear objective.

```rust
use ncalg::sdp::{Affine, SdpModel};
let mut m = SdpModel::new();
let g = m.psd_var(4, false, "G");            // Hermitian 4x4, G ⪰ 0 (16 real variables)
let w = m.free_vars(4, "w");
let c = m.free_var("c");
m.equal(&g.entry(0, 1), &Affine::constant(0.5));
m.psd_affine(4, false, (0..4).map(|a| (a, a, Affine::var(w[a]) - g.entry(a, a))).collect(), "diag(w) - G");
m.minimize(&w.iter().fold(Affine::zero(), |s, j| s + Affine::var(*j)));
m.write_json("model.json")?;                 // free variables + LMIs + equalities (CVXPY: scripts/solve_moment_json.py --out sol.txt)
let e = m.eliminate()?;                      // equalities solved out exactly
e.model.write_sdpa("model.dat-s")?;          // pure LMI form for CSDP / SDPA
let x = e.lift(&z);                          // solver's z -> original variables
m.check(&x, 1e-6).feasible();                // residuals and smallest eigenvalue
g.value(&x);                                 // the matrix at the solution
```

On top of a `Relaxation` this gives:

- `moment_equality(p, value)` / `moment_inequality(p, lower)`: linear
  constraints `L(p) = value`, `L(p) ≥ lower` on the moments, solved out
  exactly in `build()` and explicit in `build_moment_form()`; compatible with
  the symmetry reduction (the constraint must be invariant).
- `sos_model()`: the Gram-matrix form `max c  s.t.  h − c = Σ_s Σ_{a,b} G_s[a,b] u_a* g_s u_b (mod I), G_s ⪰ 0`
  (one Gram matrix per moment/localizing matrix), whose solution is an
  explicit SOS certificate (`SosModel::gram_value`, `bound_value`).
- `gram_weight_model(shift)`: the joint Gram and weight problem
  `min Σ_u w_u  s.t.  h + shift = Σ G_ab u_a* u_b, G ⪰ 0, diag(w) − G ⪰ 0`,
  the smallest total diagonal weight that dominates a Gram matrix of `h + shift`.
- Both Gram forms carry a free real multiplier `μ_j` per Hermitian part `r_j`
  of each moment constraint `p_k − v_k` (`SosModel::multipliers`,
  `multiplier_polynomials`), so the identity reads `target + Σ μ_j r_j = Σ G …`
  and is the exact dual of the constrained moment problem. The extracted `Certificate` carries the
  multiplier terms and `verify` includes them. `MomentSdp::certificate`
  (from the eliminated form's duals) has no multipliers.
- With a symmetry acting by signed permutations, both Gram forms are
  symmetry-reduced: one Gram block per block of `build()` in its orthonormal
  basis (`SosModel::gram_sources`, `gram_irrep_dimensions`), the identity
  matched on orbit representatives of its group average, multipliers for the
  group-averaged constraints, `gram_weight_model` weights constant on orbits of index
  words.  Exact; `gram_value(s, x)` lifts the blocks of source `s` and `certificate`
  averages over the group.  `MomentSdp::certificate` averages the Gram matrix
  of each source on which the group acts monomially (`G[π(a), π(b)] +=
  conj(c_a) c_b G[a, b] / |G|`) instead of the squares, so the certificate has
  `|G|` times fewer squares, and the `gram_weight_model` Gram blocks shrink by
  about the order of the group, with the same optimum.
- `MomentForm::to_model()`, `MomentSdp::to_model()` and `moment_affine(nf)`:
  turn a built relaxation into a model and express `L(p)` as an affine
  expression, so extra PSD variables or constraints can be added afterwards
  (on symmetry-reduced blocks, see the note on unitary reduction above).

`cargo run --release --example sdp_models -- --solve` builds the moment,
SOS, `gram_weight_model` and constrained models for the two-qubit Heisenberg
interaction, solves them with CVXPY/SCS and checks the solutions (moment and
SOS values both −3, `gram_weight_model` weights summing to 6).

### SOS certificates

A solved Gram model, or the dual matrices of a solved moment SDP, gives an
explicit certificate `h − c = Σ_k λ_k p_k* g_{s_k} p_k (mod I)`:

```rust
let cert = sos.certificate(&x, 1e-7);                 // from a SosModel solution x
let cert = sdp.certificate(&duals, 1e-7);             // from the dual blocks of MomentSdp (symmetry-reduced ones are lifted and averaged)
let ver = cert.verify(gb.convert::<Complex<f64>>().reducer(), &h.convert());
ver.bound;          // the constant c the identity certifies
ver.max_residual;   // what is left after reduction modulo the ideal (solver defect)
let ver = cert.verify_exact(gb.convert::<Complex<f64>>().reducer(), &h.convert());
println!("{}", cert.pruned(1e-6));                    // λ · (p)* (p) per line
```

Float coefficients vanish when a sum cancels to rounding level relative
to its summands, `|Σ c_i| ≤ 64 ε Σ |c_i|`.  Genuinely small
coefficients are kept, so `verify` reports the actual residual of a
certificate: native-solver certificates reach 1e-14 to 1e-8 (SYK N = 16–24:
≤ 1.5e-13).  An absolute floor (`set_float_tolerance`, default `0`) can be set
for data with a known noise level; `ver.float_tolerance` says which floor was
in effect.  `verify_exact` disables both the floor and the cancellation test
on the calling thread only, so only exact zeros vanish
(`ncalg::coeff::with_exact_zero` does the same for any computation, for
example `MultiSdp::certificate`).

The scripts write dual matrices on request (`solve_moment_json.py --duals
file`, `solve_sdpa.py … solution.txt duals.txt`, `solve_sdpa.jl … solution.txt
duals.txt`) in a plain block format read by `ncsos::read_dual_blocks`;
`dual_grams` accepts `m × m` Hermitian duals in the native solver's convention
(`Σ_ab G_ab F_ab = tr(G Fᵀ)`) and the real `2m × 2m` duals of real-embedded
blocks. `solve_moment_json.py` embeds complex blocks itself and writes the
real duals, so they follow ncalg's `tr(G Fᵀ)` convention directly (CVXPY's
own complex PSD duals pair as `Re tr(F Y)`, the transpose).  For the two-qubit Heisenberg
interaction the example recovers
`h + 3 = ½ Σ_{σ∈{X,Y,Z}} (σ_0 + σ_1)²` (as three squares in a rotated basis)
with residual below `1e-9`.

## Native SDP solver (`ncalg::solver`)

`SdpModel::solve` and `MomentSdp::solve` run a primal–dual interior-point
method (HKM direction, Mehrotra predictor–corrector, dense faer linear
algebra) directly on the LMI form, so the whole pipeline — Gröbner basis,
relaxation, solve, certificate — stays in Rust:

```rust
use ncalg::solver::SolverOptions;
let sdp = Relaxation::level(gb.reducer(), 2).minimize(&h).symmetry(generators).build();
let (value, sol) = sdp.solve(&SolverOptions { tol: 1e-8, ..Default::default() })?;
sol.status;                                   // Optimal | MaxIterations | NumericalError
let cert = sdp.certificate(&sol.duals, 1e-12);          // SOS certificate from the dual
let ver = cert.verify(gb.convert::<Complex<f64>>().reducer(), &h.convert());
let moments = sdp.moments(&sol.x);            // the optimal functional L
```

Equalities are eliminated internally; real symmetric blocks are solved in
real arithmetic and complex Hermitian blocks natively in complex arithmetic
(`m × m` complex matrices with `⟨F, X⟩ = Re tr(F X)`, not the `2m × 2m` real
embedding), so mixed models work and the dual returned for a complex block
is its `m × m` Hermitian Gram matrix.  Per iteration the cost is
`O(Σ n_b³ + m³)` plus the Schur-complement assembly, which picks the dense or
sparse formula per variable (Fujisawa–Kojima–Nakata), so blocks of a few
hundred and a few thousand variables are comfortable; tolerances tighter
than about `1e-9` run into the limits of double precision and end with
`NumericalError`; the solution then holds the best iterate, and `solve`
returns its value only if that iterate meets `accept_tol` (default `1e-5`),
otherwise NaN.  On tight relaxations, whose optimal moment matrix is rank
deficient, `NumericalError` with a finite value is common and accurate to
about `1e-5`.  NaN usually means the relaxation has no strictly feasible
point (every feasible moment matrix is singular).  For those:

- `SolverOptions { regularization: 1e-8, .. }` solves `Z + ε I ⪰ 0`, which
  is well posed, and returns the original problem's dual bound at the
  regularized Gram matrices: a certified bound on the correct side, looser by
  about `c √ε`.  `sol.regularization` holds a `bias_estimate` (from a second
  solve at `100 ε`) and an `extrapolated_objective` (not a bound).
- `Relaxation::eigenstate_reduction(true)` handles constraints that make the
  state an extreme eigenvector (`L(X0) = 1`, projectors at 0 or 1) exactly.
- `SolverOptions { facial_reduction: true, .. }` retries a stalled solve on
  its facial reduction when the reduction is numerically resolvable, and
  otherwise returns the unreduced result.
- A standard-form model (every LMI a bare `psd_var`, the other variables free
  and only in equalities and the objective, as in the Gram forms of
  `sos_model`) is solved through its dual LMI when that is smaller: `r`
  free variables for `r` independent equalities instead of `N − r` for `N`
  matrix entries.  The matrix variables are the dual's dual matrices, and the
  free variables come from a QR least-squares solve of the equalities.
  `SolverOptions::dualize` overrides the choice: `None` (default) decides by
  size, `Some(true)` forces the dual, `Some(false)` disables it (Heisenberg
  6-ring level-2 Gram form: 86 s → 0.4 s, and the 8- and 10-rings, refused before
  under `max_variables`, solve in 13 s and 171 s).
- Directions annihilated by every coefficient matrix of a block are always
  removed before solving (exact).  A block is first tested with a random
  combination of its coefficient matrices: if that is clearly nonsingular,
  no such direction exists, and the rank-revealing QR of the stacked
  matrices is skipped (that QR once took 57 s of a 63 s solve on 9-qubit
  Quantum Max Cut).

Every solution carries `sol.profile` (a `SolveProfile`): wall time per phase
(presolve, elimination, the dual route, Schur assembly, factorization,
Newton solves, the rest of the iterations), and counters (iterations,
operator products, factorization retries, how each block's kernel search
went).  `verbose` prints it as one line.  When one-off work takes more than a
quarter of a solve and over a second, `sol.warnings` gets a warning, printed
to stderr unless `SolverOptions::warnings` is off.  One-off work means
everything outside the interior-point iterations.  Unless that is expected
for the problem, it is a performance bug worth an issue.

Measured on a 16-thread desktop
(release build, after the speed-ups to Schur assembly, step lengths and
presolve, with the times before them in parentheses; the SYK rows time two
solves on the same parity-reduced blocks at tol 1e-8, averaged over two runs;
the complex blocks' former real embedding took 1.5–2× as long):

| problem | variables | blocks | iterations | time | value |
|---|---:|---|---:|---:|---|
| QMC 8-cycle level 2, dihedral symmetry (real) | 79 | 12, 68, 68, 17, 23, 78, 29 | 16 | 0.08 s (0.12 s) | 5.6510934 (exact) |
| QMC 8-cycle level 2, no symmetry (real) | 749 | 295 | 16 | 1.65 s (5.9 s) | 5.6510934 |
| SYK N = 8, parity-reduced, two solves (complex) | 98 | 8, 29 | – | 0.03 s (0.04 s) | match reference to 1e-8 |
| SYK N = 10, parity-reduced, two solves (complex) | 255 | 10, 46 | – | 0.11 s (0.10 s) | match reference to 1e-8 |
| SYK N = 12, parity-reduced, two solves (complex) | 561 | 12, 67 | – | 0.23 s (0.42 s) | match reference to 1e-8 |
| SYK N = 14, parity-reduced, two solves (complex) | 1092 | 14, 92 | – | 0.54 s (1.15 s) | match reference to 1e-8 |

For larger problems export to SCS / CSDP as before (`write_sdpa`,
`write_json`); the interior-point solver reaches `1e-8` where SCS at
`eps = 1e-7` is faster but less accurate.

## GNS state extraction (`ncalg::gns`)

From the optimal moments of a tight relaxation, `MomentSdp::gns` builds the
GNS representation: a Hilbert space of dimension `r = rank` of the Gram matrix
over the index words closed under left multiplication, matrices `π(x)` for
every generator and the cyclic vector `|1⟩`, with
`L(w) = ⟨1| π(w) |1⟩` for all moments (Pironio–Navascués–Acín; Henrion–Lasserre
in the commutative case, where the `r` minimizers are recovered).

```rust
let (value, sol) = sdp.solve(&SolverOptions::default())?;
let g = sdp.gns(&sol.x, gb.reducer(), 1e-6)?;      // tol: relative eigenvalue cut-off
g.dimension;                                         // 2 for the qubit, 4 for the Heisenberg pair
g.generators[v];                                     // π(x_v) as a dense complex matrix
g.expectation(&h);                                   // ⟨1| π(h) |1⟩ = value
g.flatness_defect; g.relation_defect(&rels); g.moment_defect(&sdp.moments(&sol.x));
let g = GnsRepresentation::extract(red, &moments, &basis, tol)?;   // from any moment table
```

The defects tell whether the moment matrix was actually flat (all three
near zero) or the output is only an approximation; they scale like the square
root of the solver tolerance.  Flatness holds whenever the optimal moment
matrix is unique, e.g. for Pauli / Majorana algebras at level 2, but an
interior-point solver returns the analytic center of the optimal face, so
for problems whose optimum leaves higher moments undetermined (typical for
polynomial optimization on `R^n` at low level) the extracted representation
is not exact even though the bound is.

## Parallelism

Three hot spots run on all cores through rayon (thread count from
`RAYON_NUM_THREADS`):

- **Gröbner bases**: the obstructions of the smallest pending degree are
  reduced in batches of `2 × threads` against the current basis, and the
  nonzero results are then absorbed one by one (each re-reduced against the
  elements inserted before it).  Complete bases are unchanged (the reduced
  basis is unique); `GroebnerOptions::parallel(false)` restores the
  one-pair-at-a-time loop.  Qubit swap algebra on 7 qubits: 35 s → 2.6 s on
  16 threads; 6 qubits: 1.25 s → 0.16 s.
- **Relaxation assembly**: the normal forms `NF(u_a* g u_b)` of the moment
  and localizing matrices are computed row-parallel (QMC 8-cycle level 2:
  10.4 s → 4.0 s; the remainder is the sequential elimination).
- **Interior-point solver**: the Schur complement is assembled one variable
  per task (QMC 8-cycle, symmetric: 0.72 s → 0.26 s).

Results are deterministic: indices and insertion orders never depend on the
thread schedule.

## Exact arithmetic over finite rings (`ncalg::arith`)

`arith` groups the finite-ring modules: `arith::ff`, `arith::zmod` and
`arith::abelian`, each also re-exported at the crate root (`ncalg::ff`, …).
`arith::int` holds the crate's only implementations of the integer and modular
helpers: extended gcd, modular add/sub/mul/pow/inverse, residues, primality
(deterministic Miller–Rabin) and prime factors. Plain gcd comes from
`num_integer`. New number-theoretic helpers belong there, not next to their
first caller.

## Finite fields (`ncalg::ff`)

`Gf::new(p, k)` is GF(q) for any prime power q = p^k, with p < 2^32 and q < 2^63.

- **Elements** are `u64` encodings Σ a_i p^i, where a_i is the coefficient of x^i. In characteristic 2 they are bit patterns, and addition is XOR.
- **Modulus:** by default the smallest irreducible polynomial (Rabin's test). `Gf::with_modulus` accepts a given one, for example a Conway polynomial.
- **Arithmetic:** for q ≤ 2^22, exp/log and Zech tables make add, mul and inv O(1). Larger fields use polynomial arithmetic, carry-less for p = 2.
- **Other operations:** `pow`, `frobenius`, `trace`, `norm`, `order`.

Linear algebra:

- `FqMat` (dense, over any `Gf`) has `rref`, `rank`, `nullspace`, `solve`, `inverse`, `det` and `mul`.
- `BitMat` (bit-packed GF(2), `u64` words) has the same operations, plus `dependency()`, the zero-sum finder: a nonempty set of rows summing to zero.
- `BitEchelon` is an incrementally built GF(2) echelon basis that records which inserted rows each basis row combines. `insert` returns the dependency when a row is in the span, and `reduce` returns the combination that reduces a vector. It is the one implementation of elimination with combination tracking, used by `BitMat::dependency` and by the decompositions in `pcgroup::families`.

`Fq<P, K>` is a `Coeff`/`Field` type for GF(p^k), so polynomials and Groebner bases work over any finite field.
- For `K > 1`, its tables come from a lazily built process-wide registry (`ff::field(p, k)`), and elements print as `#e` and parse back.
- For `K = 1` (the prime field, `Fp<P>`), arithmetic is direct modular arithmetic from `arith::int` with no registry lookup, and residues print as their symmetric representative. `Fq::new` takes the residue or encoding; `Coeff::from_i64` takes a signed integer.

## Groups and symmetry (`ncalg::symmetry`)

`symmetry` holds the finite-group code. Each algorithm has one implementation:

- **`symmetry::perm`:**
  - `SignedPermutation` (also at `ncsos::SignedPermutation`);
  - `Monomial`, monomial matrices with root-of-unity phases (also at `pcgroup::families::Monomial`);
  - `closure`, the one group-closure routine, behind `permutation_group` and `signed_permutation_group`.

  `SignedPermutation::compose` is the product of monomial matrices with one phase bit.
- **`symmetry::action`:** signed permutations acting modulo an ideal:
  - `image` and `word_image`, normal forms of images;
  - `average`, the Reynolds operator;
  - `preserves` and `preserves_ideal`, invariance tests;
  - `WordOrbits`, orbit classes of normal words with the coefficient of the representative in the group average.

  The relaxation builder, the reduced Gram forms and `multi` all use these.
- **`symmetry::isotypic`** (crate-internal) is the isotypic decomposition behind block-diagonalization: exact, orbit by orbit, from the group's irreducible representations for signed-permutation actions; numerical otherwise.
- **`pcgroup` and `abelian`** live here too, re-exported at `ncalg::pcgroup` and `ncalg::abelian`.

## Polycyclic groups (`ncalg::pcgroup`)

Finite p-groups given by power-conjugate presentations, with no enumeration of the group.

- **Collector.** `PcPresentation` (or `PcBuilder`) collects from the left. It provides `mul`, `inv`, `pow`, `order`, `comm` and `conjugate`, and converts between elements and integer indices with the first generator most significant (`index`, `from_index`). `check_consistency` runs the consistency test.
- **Subgroups.** `subgroup(gens)` and `normal_closure(gens)` return canonical induced pc sequences. Membership is tested by sifting (`contains`), and subgroups are equal exactly when their canonical sequences are.
- **Series.** `lower_exponent_p_central_series()` computes P_{i+1} = [G, P_i] P_i^p. `series_blocks()` checks that the pcgs refines that series and returns the layer dimensions d_i.
  - `lower_central_series()` (γ_{i+1} = [G, γ_i]), `upper_central_series()` (Z_{i+1}/Z_i = Z(G/Z_i)) and `nilpotency_class()`. `centralizer_mod(xs, m)` is {g : [g, x] ∈ M for all x}, the centralizer in G/M pulled back; the upper central series and `centralizer` are its special cases. `abelian_invariants(a, b)` and `series_factor_invariants(series)` give the cyclic decomposition of abelian factors A/B, which shows when a layer is not elementary abelian.
- **Quotients and products.** `quotient(n)` divides out any normal subgroup N and returns the projection (`PcProjection::apply`, `lift`, `kernel`). The quotient keeps G's generators at the depths where N has none; its relations are their relations projected, so N need not be spanned by pcgs generators or sit at the bottom of the series (a diagonal subgroup of a center, the terms Z_i of the upper central series). `quotient_central(positions)` is the checked special case for central subgroups spanned by generators. `direct_product(h)` concatenates the pcgs; `central_product(h, pairs)` divides G × H by {(z, φ(z)⁻¹)} for an isomorphism φ given on central generators, and checks that it is one.
- **Group algebra.** `group_algebra_relations::<C>()` gives the relations of the group algebra K[G] under the `Wreath` ordering. They are already a complete Groebner basis, with p^n normal words: Pauli-CZ(5) completes in 8 ms, where deglex did not finish in 2 min.
- **Centralizers, intersections, normalizers.** `centralizer(xs)`, `centralizer_of(s)`, `center()`, `intersection(a, b)`, `normalizer(s)`, plus `join`, `is_subgroup` and `conjugate_subgroup`. Every pcgs refines a central series with factors ℤ/p, so centralizer and intersection are one kernel computation per generator (polynomial time). The normalizer runs orbit–stabilizer along the pcgs; its cost grows with the largest index [M_j : M_{j+1}] along the series.
- **Families.** `pcgroup::families` provides `pauli_cz(m)`, `pauli_ccz(m)`, `xs(m)`, `xt(m)` and `d4n(N)`.
- **Custom groups.** `families::x_diag(name, m, shifts, phases)` builds the group generated by the shifts X^a (any independent shift basis) and the diagonal gates ω^{f(x)}, for arbitrary f: 𝔽₂^m → ℤ₈. It closes the diagonal part under conjugation and squaring, computes the central series on the phase side, and chooses a pcgs that refines it, with the scalars as separate generators. The built-in qubit families are the special cases with S, T or CZ gates. `families::x_diag_bits(name, m, shifts, phases, r)` does the same with phases mod 2^r (1 ≤ r ≤ 7, ω = e^{2πi/2^r}); `PcFamily::phase_bits()` reports r. For example r = 4 with f_j(x) = x_j gives ⟨X_j, √T_j⟩, whose projective quotient has four layers [2m, m, m, m].
  - Each comes with its monomial representation (`monomial(e)`, `decompose(mo)`), its scalar generators, and `projective()`, which gives G/⟨scalars⟩.
  - Example: XT(12), with |G| = 2^51, builds in 55 ms, and a product takes about 8 µs (`examples/pcgroup.rs`).

## Linear algebra over ℤ/N and ℤ (`ncalg::zmod`)

- `ZnMat` is a dense matrix over ℤ/N for any 2 ≤ N < 2^63. ℤ/N is not a field when N is composite, so the echelon form is the **Howell form** (`howell()`): pivots divide N, and two matrices have the same row span exactly when their Howell forms are equal. On top of it: `span_contains`, `span_factors` (|span| = Π N/pivot), `nullspace`/`left_nullspace` (generating sets) and `solve`/`left_solve`.
- `SparseHowell` is the Howell form of sparse vectors (`(column, value)` lists), built incrementally, and the one implementation behind `ZnMat::howell` and the methods above.
  - **Insertion:** `insert(v)` reduces v against the rows kept by pivot column. A leading entry the pivot divides is canceled. Otherwise a Bezout step replaces the pivot row by one with pivot gcd(a, b) and continues with the other combination. New and replaced pivot rows are normalized to divisors of N and push their annihilator rows (N/h)·r, so the rows are a Howell basis after every insertion. `insert` returns whether the span grew.
  - **Queries:** `contains(v)`, `reduce(v)` (the canonical representative of v + span), `pivots()`, `span_factors()`, `rows()` (the canonical form, reduced above the pivots, the same for any insertion order), and sparse `left_nullspace` / `left_solve`.
  - **Speed:** faster than the dense algorithm it replaced even on dense input (200×200: 0.02 s against 0.05 s). On sparse input it scales with the fill: a 1000×1000 band over ℤ/2^32 takes 8 ms against 2.3 s, a 1000×1000 matrix with 3 random entries per row takes 0.07 s against 2.0 s, and 3000×3000 bands take 9 ms.
- `IntMat` is a dense `BigInt` matrix with `hermite()` (H = U A), `smith()` (U A V = D, d_0 | d_1 | ⋯), `kernel()` and `solve()`.

## Finite abelian groups (`ncalg::abelian`)

`AbelianGroup::new(vec![d_0, …])` is ℤ/d_0 × ⋯; `from_relations(k, R)` gives ℤ^k / ⟨rows of R⟩ in invariant-factor form, with the images of the generators.

- **Subgroups** are stored as the Howell form of their image in (ℤ/e)^k, e the exponent, so they compare with `==`. Operations: `subgroup`, `contains`, `subgroup_order`, `join`, `intersection`, `annihilator` (H^⊥ in the dual), `quotient`, `invariants`.
- **Duality.** The dual is identified with G by χ_y(x) = exp(2πi Σ x_i y_i / d_i) (`pairing`, `character`).
- **Fourier transform.** `fourier(&mut f, inverse)` is the unitary transform of a function stored by `index`, one pass per cyclic factor (radix 2 for power-of-two factors).
- **Zero-sum subsequences.** `zero_sum_subset(seq, strategy)` returns a nonempty set of indices whose elements sum to zero, computed from the elements alone (deterministic):
  - `ZeroSum::Linear` for 𝔽₂^d: Gaussian elimination (`BitMat::dependency`); complete, d + 1 elements always suffice;
  - `ZeroSum::Leveled` for 2-groups ⊕ ℤ/2^{e_i}: subsets with even sum, halved and recursed; succeeds on Π_ℓ (d_ℓ + 1) elements (`zero_sum_length`), d_ℓ = #{i : e_i ≥ ℓ}; polynomial;
  - `ZeroSum::Exhaustive` for any group: smallest, then lexicographically first; exponential.

  `davenport_bound()` gives D(G) exactly for p-groups and rank ≤ 2 (D*(G) = 1 + Σ(a_i − 1), Olson; van Emde Boas–Kruyswijk), otherwise the upper bound exp(G)(1 + ln(|G|/exp(G))), with an `exact` flag.

## Cyclotomic numbers (`ncalg::cyclo`)

`Cyclo<N>` is ℚ(ζ_N) as a `Coeff` and `Field` (and `ComplexCoeff` when 4 | N), stored in the power basis modulo Φ_N, so equality is exact.

- **Operations:** `zeta(k)`, `sqrt2()`, `galois(a)`, `norm()`, `conj`, `real_part`, `abs_sq`, `to_complex`.
- **Exact comparison of real elements:** `sign()` and `cmp_real()`. A real x equals g(cos 2π/N) for a Chebyshev combination g. The root cos 2π/N of U_{N−1} is isolated with certified sign changes, then bisected in rational arithmetic until interval evaluation of g excludes zero. Use this to certify inequalities such as cos(π/8) > 1/√2.
- **Literals:** rationals, `i` (when 4 | N) and `#k` for ζ^k; elements print as sums of `c*#k` and parse back.
- **Exponential sums (`cyclo::expsum`).** `PhasePolynomial` is f: 𝔽₂ⁿ → ℤ/2^r as c + Σ c_T Π [λ·x], products of parities (monomials are the case of single variables). `exp_sum::<N>()` and `exp_sum_over::<N>(a, basis)` compute Σ_{x∈a+V} ζ^{f(x)} exactly in ℚ(ζ_N); `restrict(a, basis)` substitutes x = a + Mt, which only maps parities to parities (in monomials it is not closed: x₁ = t₁⊕t₂⊕t₃ makes a T phase cubic).
  - **Clifford class** (every term of level ≤ 2: S/Z linear, CZ quadratic): ζ^f = ζ^c i^{q(x)} with q a ℤ₄ quadratic form, summed by Gauss-sum elimination in O(n³/64); `clifford_sum()` gives the closed form 0 or √2^p e^{2πiφ}, so |S|² ∈ {0} ∪ 2^ℤ. 300 variables on a 250-dimensional subspace take milliseconds.
  - **Everything else is exact, exponential only in a width.** The non-Clifford parities are pivoted onto d = their rank coordinates, all other directions are eliminated as above, and the core is summed by variable elimination over exact cyclotomic integers. `exp_sum_plan()` reports the core size and the width w, and the cost is poly + O(m 2^w). Polynomial for product-type instances (T on every qubit of |+⟩ⁿ: w = 1, 200 qubits in 5 ms), a few non-Clifford gates (w ≤ d), and bounded treewidth (a CS chain or ring: w ≤ 3).
  - **Why not more:** by the Cai–Lu–Xia dichotomy for complex-weighted Boolean #CSP, these sums are polynomial exactly for affine functions (the Clifford class) and product-type ones, and #P-hard for any other set of phase types (T, CS = i^{xy}, CCZ with CZ couplings or general subspaces).

## Built-in relation sets

`ncalg::relations` provides generators for common algebras, each returning the
algebra (with a sensible generator order) and the relations:

| function | algebra |
|---|---|
| `commutative(&alg)` | polynomial ring |
| `q_commutative(&alg, q)` | quantum affine space |
| `weyl(n)` | Weyl algebra `A_n` on `x_i, d_i` |
| `lie_algebra(&alg, bracket)`, `sl2()`, `so3()` | universal enveloping algebras (relations are a Gröbner basis iff Jacobi holds — PBW) |
| `pauli(n)` | `n`-qubit Pauli algebra over complex coefficients (`4^n` normal words) |
| `majorana(n)`, `clifford(signs)`, `exterior(n)` | Clifford-type algebras (`2^n` normal words) |
| `symmetric_group(n)`, `symmetric_group_3()` | Coxeter presentations of `S_n` (`n!` normal words) |

## Examples and benchmarks

```
cargo run --example pauli                        # 2-qubit Pauli algebra walk-through
cargo run --example partial_basis                # infinite basis, truncated and resumed
cargo run --release --example reduce_while_expanding
cargo run --release --example bench              # timings of all the standard cases
cargo run --release --example stress [pauli|sym|sl3|random]
cargo run --release --example custom_algebra [-- --free]   # hand-built *-algebra relaxation (QAOA on a path)
cargo run --release --example multi_quench       # linked functionals: a one-qubit quench, both sides, certified
python scripts/canary.py                         # performance canary against this machine's baseline
```

`scripts/canary.py` runs `examples/canary.rs`: one instance per structural
regime, about 15 s in all.  The regimes cover dense-overlap coefficient
matrices, disjoint supports (Pauli and Majorana), projectors, commutative
variables, one large block with rank-one constraints, many constraints on a
small block, and symmetry-reduced blocks.  Each run records the build, solve
and certificate times, the solver's profile and its counters, and compares
them with this machine's baseline (`canary/baseline-<host>.json`, written
locally and not committed).  See CONTRIBUTING.md (Performance).

`examples/custom_algebra.rs` is the template for a relaxation in an algebra
of your own: Pauli-type relations plus central scalar generators, a
structured index set, vector-form constraints `L(a* R b) = 0` filtered by
`constraint_coverage()`, a symmetry with `symmetrize_constraints`, and the
choice between the native solver and the moment form by size.

## Testing

`cargo test` runs about 280 tests, among them: unit tests for coefficients, words,
orderings, the automaton, the parser and the heap reducer; integration tests
checking Gröbner bases against known results (PBW for `sl_2`, `so_3`, Weyl
algebras, Pauli/Clifford/exterior algebras and symmetric groups by dimension
count, the classical infinite basis of `<xyx − xy>`), the equivalence of lazy
and eager reduction on random inputs, uniqueness of normal forms, ideal
membership of random ideal elements, equality of results with and without the
chain criterion, degree-truncation guarantees, and every termination mode.

## Bugs and improvements

Known problems and planned work are in
[GitHub Issues](https://github.com/aidan-epperly/ncalg/issues).  File new ones
with the issue forms there, or from the command line with `python
scripts/issues.py new bug "<title>" --area <area> --severity <1|2|3>`, which
writes a file to [`issues/`](issues/README.md) and mirrors it to GitHub.  The
workflow and severities are described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Design notes and limitations

- Coefficient fields are required for Gröbner bases and reduction; integer
  input is promoted to `BigRational`.  Strong Gröbner bases over `Z` are not
  implemented.
- Floating point arithmetic decides cancellation relative to the summands
  (`|Σ c_i| ≤ 64 ε Σ |c_i|`, scale-invariant).  A Gröbner basis *computed* in `f64` or
  `Complex<f64>` is not backward stable: the term order, not the size of the
  coefficients, fixes every pivot (`make_monic`), so rounding errors can be
  amplified without bound, and for general noncommutative ideals no
  comparably stable algorithm is known.  Compute over `BigRational` (or
  `Fp<P>` for leading words and Hilbert series) and convert the basis with
  `GroebnerBasis::convert::<f64>()`; reduction against a converted exact
  basis is fine.  Float Gröbner computations are for small, well-scaled
  examples only.
- The divisor chosen during reduction is the one whose occurrence ends
  earliest in the word.  For a complete Gröbner basis the normal form is
  independent of this choice.
- The chain criterion relies on the active basis being interreduced (distinct
  leading words, none a subword of another), which the solver maintains.
- Words are `Vec<u32>` and every stream term allocates its concatenated word;
  this keeps the code simple and safe at the cost of some allocation pressure
  on very large computations.
- Words are compared through a trait object (`dyn MonomialOrder`); the
  comparison itself is `O(length)`, so the indirection is a small constant.

## Elimination of moment constraints

`Relaxation::build` solves the moment constraints out of the LMI
exactly: with `y = B z` the moments after Hermiticity and symmetry and
`R y = 0` the constraints, it needs the kernel of `C = R B` with the unit
functional surviving.  Floating-point constraint systems from physics
(recursion identities between moments, `fixtures/constraint_elimination/`) are *exactly*
inconsistent with `L(1) = 1` as rational matrices while being consistent to
about `1e-10`, so the object needed is the near-kernel
`K_δ = span{v_i : σ_i ≤ δ‖C‖}` at the noise level, not the exact kernel.
Gaussian elimination cannot find it (its oblique remainders are unrelated to
the distance of a row from the span of the earlier rows), which is why the
constraint step uses `ncalg::sparse_qr`: a row-sequential sparse Givens QR
with Heath's rank detection.  Rotations preserve
the Gram matrix, so every kernel vector `k` satisfies
`‖C k‖ ≤ η ‖k‖` with `η` the norm of the discarded remainders; the only
possible failure is a too-high rank (Kahan-type matrices), which the
verification `C K ≈ 0` and the normalization check report as an error with
numbers instead of a wrong SDP.  `Relaxation::elimination_tolerance` (default
`1e-9`, overridable with `NCALG_ELIMINATION_TOL`) is the rank tolerance;
`NCALG_DEBUG_ELIM=1` prints ranks, fill and residuals; `NCALG_DUMP_CONSTRAINTS=<prefix>`
writes `C` and the unit functional as Matrix Market fixtures.  On the
22 716 × 31 744 fixture (`elim_large`): rank 18 425, 13 319 variables left, 0.8 M nonzeros
in `R`, 3.9 s; the old elimination declared it inconsistent.  The structural
eliminations (Hermiticity, symmetry; `SdpModel::eliminate`) keep the exact
sparse Gaussian elimination, whose data are exact small rationals.

## License

Copyright (C) 2026 Aidan Epperly.  Licensed under the GNU General Public
License, version 3 or (at your option) any later version; see [LICENSE](LICENSE).
