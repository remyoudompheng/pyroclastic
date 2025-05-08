"""
Extra bindings for Python-flint
"""

cdef extern from "gmp.h":
    ctypedef unsigned long ulong
    ctypedef unsigned long mp_limb_t
    ctypedef unsigned long mp_bitcnt_t

cdef extern from "flint/fmpz.h":
    ctypedef long slong
    ctypedef ulong flint_bitcnt_t
    ctypedef ulong * nn_ptr

cdef extern from "flint/flint.h":
    ctypedef struct nmod_t:
        mp_limb_t n
        mp_limb_t ninv
        flint_bitcnt_t norm

cdef extern from "flint/nmod_types.h":
    ctypedef struct nmod_poly_struct:
        nn_ptr coeffs
        slong alloc
        slong length
        nmod_t mod

    ctypedef nmod_poly_struct nmod_poly_t[1]

    ctypedef struct nmod_berlekamp_massey_struct:
        slong npoints
        nmod_poly_t R0
        nmod_poly_t R1
        nmod_poly_t V0
        nmod_poly_t V1
        nmod_poly_t qt
        nmod_poly_t rt
        nmod_poly_t points

    ctypedef nmod_berlekamp_massey_struct nmod_berlekamp_massey_t[1]

cdef extern from "flint/nmod_poly.h":
    slong nmod_poly_degree(const nmod_poly_t poly)
    ulong nmod_poly_get_coeff_ui(const nmod_poly_t poly, slong j)

    void nmod_berlekamp_massey_init(nmod_berlekamp_massey_t B, ulong p)
    void nmod_berlekamp_massey_clear(nmod_berlekamp_massey_t B)
    void nmod_berlekamp_massey_start_over(nmod_berlekamp_massey_t B)
    void nmod_berlekamp_massey_set_prime(nmod_berlekamp_massey_t B, ulong p)
    void nmod_berlekamp_massey_add_points(nmod_berlekamp_massey_t B, const ulong * a, slong count)
    void nmod_berlekamp_massey_add_zeros(nmod_berlekamp_massey_t B, slong count)
    void nmod_berlekamp_massey_add_point(nmod_berlekamp_massey_t B, ulong a)
    int nmod_berlekamp_massey_reduce(nmod_berlekamp_massey_t B)
    slong nmod_berlekamp_massey_point_count(const nmod_berlekamp_massey_t B)
    const ulong * nmod_berlekamp_massey_points(const nmod_berlekamp_massey_t B)
    const nmod_poly_struct * nmod_berlekamp_massey_V_poly(const nmod_berlekamp_massey_t B)
    const nmod_poly_struct * nmod_berlekamp_massey_R_poly(const nmod_berlekamp_massey_t B)

def berlekamp_massey(seq: list, p: int) -> list:
    cdef nmod_berlekamp_massey_t B
    cdef ulong pi = p
    cdef ulong xi
    nmod_berlekamp_massey_init(B, pi)
    for x in seq:
        xi = x
        nmod_berlekamp_massey_add_point(B, xi)
    nmod_berlekamp_massey_reduce(B)

    cdef nmod_poly_struct *v = nmod_berlekamp_massey_V_poly(B)
    cdef slong deg = nmod_poly_degree(v)
    res = []
    for i from 0 <= i <= deg:
        res.append(nmod_poly_get_coeff_ui(v, i))
    nmod_berlekamp_massey_clear(B)
    return res
