//! Saving and loading polynomials and Gröbner bases.
//!
//! The format is plain UTF-8 text, one polynomial per line in the same
//! syntax the parser accepts, preceded by a header that records everything
//! needed to rebuild the algebra:
//!
//! ```text
//! ncalg v1
//! kind: groebner-basis            (or: polynomials)
//! generators: x, y, a, a†
//! order: deglex                   (any MonomialOrder::name(), e.g. degrevlex, wdeglex([1, 2]), elim(1))
//! adjoint: 0 1 3 2                (involution on generator indices)
//! coefficients: BigRational       (type name, checked on load)
//! termination: complete           (groebner-basis only)
//! count: 3
//! x*y - y*x
//! …
//! ```
//!
//! Lines starting with `#` are comments and are ignored.  Files written by
//! [`GroebnerBasis::save`] are therefore also readable by hand and by other
//! tools, and the examples' ad hoc dumps can be converted by adding a header.

use crate::algebra::FreeAlgebra;
use crate::coeff::{Coeff, Field};
use crate::groebner::{GroebnerBasis, Termination};
use crate::order::parse_order;
use crate::poly::Polynomial;
use crate::word::Var;
use std::fmt::Write as _;
use std::io;
use std::path::Path;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Header describing an algebra and a list of polynomials.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    /// `groebner-basis` or `polynomials`.
    pub kind: String,
    /// Generator names, in index order.
    pub generators: Vec<String>,
    /// The monomial ordering, as `MonomialOrder::name()`.
    pub order: String,
    /// The involution on generator indices.
    pub adjoint: Vec<Var>,
    /// Coefficient type name, checked on load.
    pub coefficients: String,
    /// Why the computation stopped (Gröbner bases only).
    pub termination: Option<Termination>,
    /// Number of polynomials that follow.
    pub count: usize,
}

/// Textual name of a coefficient type, as written in the header.
pub fn coefficient_type_name<C: Coeff>() -> String {
    let full = std::any::type_name::<C>();
    // strip module paths: "num_rational::Ratio<num_bigint::BigInt>" -> "Ratio<BigInt>"
    let mut out = String::new();
    let mut ident = String::new();
    for ch in full.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            ident.push(ch);
        } else {
            if !ident.is_empty() {
                out.push_str(&ident);
                ident.clear();
            }
            if ch == ':' {
                // drop the accumulated path segment
                while let Some(c) = out.pop() {
                    if !(c.is_alphanumeric() || c == '_') {
                        out.push(c);
                        break;
                    }
                }
            } else {
                out.push(ch);
            }
        }
    }
    out.push_str(&ident);
    out.replace("Ratio<BigInt>", "BigRational")
}

fn write_header(
    s: &mut String,
    kind: &str,
    alg: &FreeAlgebra,
    coefficients: &str,
    termination: Option<&Termination>,
    count: usize,
) {
    let _ = writeln!(s, "ncalg v1");
    let _ = writeln!(s, "kind: {}", kind);
    let _ = writeln!(s, "generators: {}", alg.names().join(", "));
    let _ = writeln!(s, "order: {}", alg.order().name());
    let adj: Vec<String> = alg.adjoint_map().iter().map(|v| v.to_string()).collect();
    let _ = writeln!(s, "adjoint: {}", adj.join(" "));
    let _ = writeln!(s, "coefficients: {}", coefficients);
    if let Some(t) = termination {
        let _ = writeln!(s, "termination: {}", termination_to_string(t));
    }
    let _ = writeln!(s, "count: {}", count);
}

fn termination_to_string(t: &Termination) -> String {
    match t {
        Termination::Complete => "complete".to_string(),
        Termination::DegreeBound {
            bound,
            smallest_pending,
        } => format!("degree-bound {} {}", bound, smallest_pending),
        Termination::PairLimit => "pair-limit".to_string(),
        Termination::BasisSizeLimit => "basis-size-limit".to_string(),
        Termination::TimeLimit => "time-limit".to_string(),
        Termination::Canceled => "canceled".to_string(),
        Termination::NotStarted => "not-started".to_string(),
    }
}

fn termination_from_string(s: &str) -> io::Result<Termination> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    match parts.as_slice() {
        ["complete"] => Ok(Termination::Complete),
        ["degree-bound", b, p] => Ok(Termination::DegreeBound {
            bound: b.parse().map_err(|_| invalid("bad degree bound"))?,
            smallest_pending: p.parse().map_err(|_| invalid("bad pending degree"))?,
        }),
        ["pair-limit"] => Ok(Termination::PairLimit),
        ["basis-size-limit"] => Ok(Termination::BasisSizeLimit),
        ["time-limit"] => Ok(Termination::TimeLimit),
        // also accept the British spelling "cancelled"
        ["canceled"] | ["cancelled"] => Ok(Termination::Canceled),
        ["not-started"] => Ok(Termination::NotStarted),
        _ => Err(invalid(format!("unknown termination {:?}", s))),
    }
}

/// Parse the header and return it together with the remaining (polynomial) lines.
pub fn parse_header(text: &str) -> io::Result<(Header, Vec<&str>)> {
    let mut lines = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty());
    let first = lines.next().ok_or_else(|| invalid("empty file"))?;
    if first.trim() != "ncalg v1" {
        return Err(invalid(format!("not an ncalg v1 file (first line {:?})", first)));
    }
    let mut header = Header {
        kind: String::new(),
        generators: Vec::new(),
        order: "deglex".to_string(),
        adjoint: Vec::new(),
        coefficients: String::new(),
        termination: None,
        count: 0,
    };
    let mut rest: Vec<&str> = Vec::new();
    let mut in_header = true;
    for line in lines {
        if in_header {
            if let Some((key, value)) = line.split_once(':') {
                let value = value.trim();
                match key.trim() {
                    "kind" => header.kind = value.to_string(),
                    "generators" => {
                        header.generators = value
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    }
                    "order" => header.order = value.to_string(),
                    "adjoint" => {
                        header.adjoint = value
                            .split_whitespace()
                            .map(|t| t.parse::<Var>().map_err(|_| invalid("bad adjoint index")))
                            .collect::<io::Result<Vec<Var>>>()?
                    }
                    "coefficients" => header.coefficients = value.to_string(),
                    "termination" => header.termination = Some(termination_from_string(value)?),
                    "count" => {
                        header.count = value.parse().map_err(|_| invalid("bad count"))?;
                        in_header = false;
                    }
                    other => return Err(invalid(format!("unknown header field {:?}", other))),
                }
                continue;
            }
            return Err(invalid(format!("malformed header line {:?}", line)));
        }
        rest.push(line);
    }
    if header.generators.is_empty() {
        return Err(invalid("header has no generators"));
    }
    if header.adjoint.is_empty() {
        header.adjoint = (0..header.generators.len() as Var).collect();
    }
    Ok((header, rest))
}

/// Rebuild the algebra described by a header.
pub fn algebra_from_header(h: &Header) -> io::Result<FreeAlgebra> {
    let order =
        parse_order(&h.order).ok_or_else(|| invalid(format!("unknown monomial order {:?}", h.order)))?;
    Ok(FreeAlgebra::with_boxed_order(
        h.generators.clone(),
        order,
        h.adjoint.clone(),
    ))
}

/// A generator whose name is also a coefficient literal of `C` (`i` over
/// complex coefficients): the text form cannot tell them apart, and the
/// parser reads the generator, so `i·j` written as `i*j` came back as the
/// word `i j`.
fn literal_named_generator<C: Coeff>(alg: &FreeAlgebra) -> Option<&str> {
    alg.names()
        .iter()
        .map(|n| n.as_str())
        .find(|n| C::parse_literal(n).is_some())
}

/// Serialize polynomials with a header.
///
/// Panics if a generator's name is also a coefficient literal of `C` (such
/// as `i` over complex numbers): the text could not be read back faithfully.
pub fn polynomials_to_string<C: Coeff>(alg: &FreeAlgebra, polys: &[Polynomial<C>]) -> String {
    if let Some(name) = literal_named_generator::<C>(alg) {
        panic!(
            "cannot serialize: the generator {:?} reads as a coefficient of type {}; rename it",
            name,
            coefficient_type_name::<C>()
        );
    }
    let mut s = String::new();
    write_header(
        &mut s,
        "polynomials",
        alg,
        &coefficient_type_name::<C>(),
        None,
        polys.len(),
    );
    for p in polys {
        alg.check_same(p.algebra());
        let _ = writeln!(s, "{}", p);
    }
    s
}

/// Save polynomials (all in `alg`) to a file.
pub fn save_polynomials<C: Coeff>(
    path: impl AsRef<Path>,
    alg: &FreeAlgebra,
    polys: &[Polynomial<C>],
) -> io::Result<()> {
    std::fs::write(path, polynomials_to_string(alg, polys))
}

/// Parse polynomials and their algebra from text produced by
/// [`polynomials_to_string`] or [`groebner_basis_to_string`].  The
/// coefficient type must match the header (so that e.g. an exact basis is not
/// silently read as floats); use [`load_polynomials_as`] to override.
pub fn polynomials_from_str<C: Coeff>(text: &str) -> io::Result<(Header, FreeAlgebra, Vec<Polynomial<C>>)> {
    let (header, lines) = parse_header(text)?;
    let expected = coefficient_type_name::<C>();
    if header.coefficients != expected {
        return Err(invalid(format!(
            "coefficient type mismatch: file has {}, requested {} (use load_polynomials_as to convert)",
            header.coefficients, expected
        )));
    }
    read_body(header, lines)
}

/// Like [`polynomials_from_str`] but ignoring the recorded coefficient type
/// (the literals are re-parsed as `C`; exact rationals become floats, etc.).
pub fn polynomials_from_str_as<C: Coeff>(
    text: &str,
) -> io::Result<(Header, FreeAlgebra, Vec<Polynomial<C>>)> {
    let (header, lines) = parse_header(text)?;
    read_body(header, lines)
}

fn read_body<C: Coeff>(
    header: Header,
    lines: Vec<&str>,
) -> io::Result<(Header, FreeAlgebra, Vec<Polynomial<C>>)> {
    let alg = algebra_from_header(&header)?;
    if let Some(name) = literal_named_generator::<C>(&alg) {
        return Err(invalid(format!(
            "the generator {:?} reads as a coefficient of type {}: the polynomials are ambiguous",
            name,
            coefficient_type_name::<C>()
        )));
    }
    if lines.len() != header.count {
        return Err(invalid(format!(
            "header announces {} polynomials, file has {}",
            header.count,
            lines.len()
        )));
    }
    let mut polys = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let p: Polynomial<C> = alg
            .parse(line)
            .map_err(|e| invalid(format!("polynomial {} ({:?}): {}", i + 1, line, e)))?;
        polys.push(p);
    }
    Ok((header, alg, polys))
}

/// Load polynomials and their algebra from a file.
pub fn load_polynomials<C: Coeff>(path: impl AsRef<Path>) -> io::Result<(FreeAlgebra, Vec<Polynomial<C>>)> {
    let text = std::fs::read_to_string(path)?;
    let (_, alg, polys) = polynomials_from_str::<C>(&text)?;
    Ok((alg, polys))
}

/// Load polynomials, converting the coefficient literals to `C` regardless of
/// the recorded type.
pub fn load_polynomials_as<C: Coeff>(
    path: impl AsRef<Path>,
) -> io::Result<(FreeAlgebra, Vec<Polynomial<C>>)> {
    let text = std::fs::read_to_string(path)?;
    let (_, alg, polys) = polynomials_from_str_as::<C>(&text)?;
    Ok((alg, polys))
}

/// Serialize a Gröbner basis with its status.
pub fn groebner_basis_to_string<C: Field>(gb: &GroebnerBasis<C>) -> String {
    if let Some(name) = literal_named_generator::<C>(gb.algebra()) {
        panic!(
            "cannot serialize: the generator {:?} reads as a coefficient of type {}; rename it",
            name,
            coefficient_type_name::<C>()
        );
    }
    let mut s = String::new();
    write_header(
        &mut s,
        "groebner-basis",
        gb.algebra(),
        &coefficient_type_name::<C>(),
        Some(gb.termination()),
        gb.len(),
    );
    for g in gb.elements() {
        let _ = writeln!(s, "{}", g);
    }
    s
}

/// Parse a Gröbner basis saved with [`groebner_basis_to_string`].  No
/// recomputation happens; the elements are trusted (use
/// [`GroebnerBasis::verify`] to check).
///
/// A file without a `termination:` line is not trusted as complete
/// (`Termination::NotStarted`): a hand-written basis used to be, which
/// turns on normal-form uniqueness shortcuts downstream.
pub fn groebner_basis_from_str<C: Field>(text: &str) -> io::Result<GroebnerBasis<C>> {
    let (header, alg, polys) = polynomials_from_str::<C>(text)?;
    if header.kind != "groebner-basis" {
        return Err(invalid(format!(
            "file kind is {:?}, not a Gröbner basis",
            header.kind
        )));
    }
    let termination = header.termination.unwrap_or(Termination::NotStarted);
    Ok(GroebnerBasis::from_elements(&alg, polys, termination))
}

impl<C: Field> GroebnerBasis<C> {
    /// Save the basis (algebra, ordering, involution, status, elements) to a text file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        std::fs::write(path, groebner_basis_to_string(self))
    }

    /// Load a basis saved with [`save`](Self::save).  The coefficient type must match.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        groebner_basis_from_str::<C>(&text)
    }

    /// Load a basis converting the coefficient literals to `C` (e.g. a
    /// rational basis read as `f64` for numerical work).
    pub fn load_as(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let (header, alg, polys) = polynomials_from_str_as::<C>(&text)?;
        let termination = header.termination.unwrap_or(Termination::NotStarted);
        Ok(GroebnerBasis::from_elements(&alg, polys, termination))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;

    #[test]
    fn type_names() {
        assert_eq!(coefficient_type_name::<f64>(), "f64");
        assert_eq!(coefficient_type_name::<BigRational>(), "BigRational");
        assert_eq!(
            coefficient_type_name::<num_complex::Complex<f64>>(),
            "Complex<f64>"
        );
        assert_eq!(
            coefficient_type_name::<num_complex::Complex<BigRational>>(),
            "Complex<BigRational>"
        );
    }
}
