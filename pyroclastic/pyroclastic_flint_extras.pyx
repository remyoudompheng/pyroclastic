"""
Extra bindings for Python-flint
"""

import flint
from libc.stdlib cimport free

cdef extern from "gmp.h":
    ctypedef unsigned long ulong
    ctypedef unsigned long mp_limb_t
    ctypedef unsigned long mp_bitcnt_t

cdef extern from "flint/fmpz.h":
    ctypedef long slong
    ctypedef ulong flint_bitcnt_t
    ctypedef ulong * nn_ptr

ctypedef long fmpz_struct
ctypedef fmpz_struct fmpz_t[1]

cdef extern from "flint/fmpz.h":
    void fmpz_init(fmpz_t f)
    void fmpz_clear(fmpz_t f)
    char * fmpz_get_str(char * str, int b, const fmpz_t f)
    void fmpz_abs(fmpz_t f1, const fmpz_t f2)
    int fmpz_root(fmpz_t r, const fmpz_t f, slong n)

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

cdef extern from "flint/qfb.h":
    ctypedef struct qfb_struct:
        fmpz_t a
        fmpz_t b
        fmpz_t c

    ctypedef qfb_struct qfb_t[1]

    void qfb_init(qfb_t q)
    void qfb_clear(qfb_t q)
    void qfb_discriminant(fmpz_t D, qfb_t f)
    void qfb_reduce(qfb_t r, qfb_t f, fmpz_t D)
    void qfb_nucomp(qfb_t r, const qfb_t f, const qfb_t g, fmpz_t D, fmpz_t L)
    void qfb_pow_ui(qfb_t r, qfb_t f, fmpz_t D, ulong exp)
    void qfb_pow(qfb_t r, qfb_t f, fmpz_t D, fmpz_t exp)
    void qfb_inverse(qfb_t r, qfb_t f)
    void qfb_principal_form(qfb_t f, fmpz_t D)
    int qfb_is_primitive(qfb_t f)
    void qfb_prime_form(qfb_t r, fmpz_t D, fmpz_t p)

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

cdef class fmpz:
    cdef fmpz_t val

cdef class qfb:
    cdef qfb_t val

    def __cinit__(self):
        qfb_init(self.val)

    def __dealloc__(self):
        qfb_clear(self.val)

    def __str__(self):
        return f"qfb{self.q()}"

    def __repr__(self):
        return f"qfb{self.q()}"

    def q(self):
        cdef char *cstr
        cstr = fmpz_get_str(NULL, 16, self.val.a)
        a = int((<bytes>cstr).decode(), 16)
        free(cstr)

        bstr = fmpz_get_str(NULL, 16, self.val.b)
        b = int((<bytes>cstr).decode(), 16)
        free(cstr)

        cstr = fmpz_get_str(NULL, 16, self.val.c)
        c = int((<bytes>cstr).decode(), 16)
        free(cstr)

        return (a, b, c)

    def __mul__(qfb self, other):
        if not isinstance(other, qfb):
            raise TypeError(repr(other))

        assert qfb_is_primitive(self.val), self

        cdef fmpz_t D
        cdef fmpz_t L
        cdef qfb q2 = <qfb>other
        cdef qfb res = qfb.__new__(qfb)
        fmpz_init(L)
        fmpz_init(D)
        qfb_discriminant(D, self.val)
        fmpz_abs(L, D)
        fmpz_root(L, L, 4)
        qfb_nucomp(res.val, self.val, q2.val, D, L)
        qfb_reduce(res.val, res.val, D)
        fmpz_clear(D)
        fmpz_clear(L)
        return res

    def __pow__(qfb self, exp):
        cdef qfb res
        cdef fmpz_t D
        res = qfb.__new__(qfb)
        fmpz_init(D)
        qfb_discriminant(D, self.val)
        cdef fmpz e
        cdef fmpz_t eabs
        neg = exp < 0
        if isinstance(exp, int):
            e = <fmpz>flint.fmpz(exp)
        elif isinstance(exp, flint.fmpz):
            e = <fmpz>exp
        else:
            raise TypeError(repr(exp))
        if neg:
            # qfb_pow requires positive exponent
            fmpz_abs(eabs, e.val)
            qfb_pow(res.val, self.val, D, eabs)
            qfb_inverse(res.val, res.val)
            fmpz_clear(eabs)
        else:
            qfb_pow(res.val, self.val, D, e.val)
        fmpz_clear(D)
        return res

    @classmethod
    def prime_form(cls, D, p):
        if isinstance(D, flint.fmpz):
            Dz = D
        else:
            Dz = flint.fmpz(D)
        if isinstance(p, flint.fmpz):
            pz = p
        else:
            pz = flint.fmpz(p)
        assert isinstance(Dz, flint.fmpz)
        assert isinstance(pz, flint.fmpz)
        q = qfb()
        if p == 1:
            qfb_principal_form(q.val, (<fmpz>Dz).val)
        else:
            qfb_prime_form(q.val, (<fmpz>Dz).val, (<fmpz>pz).val)
        return q
