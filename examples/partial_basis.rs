//! An ideal with an infinite Gröbner basis, computed up to a degree bound and
//! then resumed, plus a time-limited run.
//!
//! `cargo run --example partial_basis`

use ncalg::{BigRational, FreeAlgebra, GroebnerOptions, GroebnerSolver, Polynomial, Termination};
use std::time::Duration;

fn main() {
    let alg = FreeAlgebra::new(["x", "y"]);
    // <x y x - x y> has the infinite reduced Gröbner basis { x y^n x - x y^n : n >= 1 }
    let f: Polynomial<BigRational> = alg.parse("x*y*x - x*y").unwrap();

    let mut solver = GroebnerSolver::new(
        vec![f.clone()],
        GroebnerOptions::new().max_degree(6).verbose(false),
    );
    let status = solver.run();
    println!("first run: {}", status);
    let gb = solver.basis();
    println!("{}", gb);
    println!("pending obstructions: {}", solver.pending_pairs());
    println!(
        "smallest pending overlap degree: {:?}",
        solver.min_pending_degree()
    );

    // the partial basis still gives correct normal forms up to its degree
    let p: Polynomial<BigRational> = alg.parse("x*y*x*y*x*y").unwrap();
    println!("\nNF(x y x y x y) = {}", gb.reduce(&p));
    let big: Polynomial<BigRational> = alg.parse("x*y^6*x - x*y^6").unwrap();
    println!(
        "x y^6 x - x y^6 in ideal according to partial basis? {}",
        gb.contains(&big)
    );

    // resume with a larger bound: earlier work is reused
    solver.set_options(GroebnerOptions::new().max_degree(12));
    let status = solver.run();
    println!("\nsecond run: {}", status);
    let gb = solver.basis();
    println!(
        "{} elements now; x y^6 x - x y^6 in ideal? {}",
        gb.len(),
        gb.contains(&big)
    );
    println!(
        "brute-force verification up to degree 12: {} failing obstructions",
        gb.verify(Some(12)).len()
    );

    // wall-clock limit on an unbounded run
    let mut solver = GroebnerSolver::new(
        vec![f],
        GroebnerOptions::new().time_limit(Duration::from_millis(100)),
    );
    let status = solver.run();
    match &status {
        Termination::TimeLimit => println!(
            "\ntime-limited run stopped with {} elements",
            solver.basis().len()
        ),
        other => println!("\nunexpected: {}", other),
    }
    println!("stats: {:?}", solver.stats());
}
