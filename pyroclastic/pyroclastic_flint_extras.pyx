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

cdef extern from "flint/flint.h":
    ctypedef struct nmod_t:
        mp_limb_t n
        mp_limb_t ninv
        flint_bitcnt_t norm

ctypedef long fmpz_struct
ctypedef fmpz_struct fmpz_t[1]

cdef extern from "flint/fmpz.h":
    void fmpz_init(fmpz_t f)
    void fmpz_clear(fmpz_t f)
    char * fmpz_get_str(char * str, int b, const fmpz_t f)
    void fmpz_abs(fmpz_t f1, const fmpz_t f2)
    int fmpz_root(fmpz_t r, const fmpz_t f, slong n)

cdef extern from "flint/fmpz_mod_types.h":
    ctypedef struct fmpz_mod_ctx_struct:
        fmpz_t n
        void * add_fxn
        void * sub_fxn
        void * mul_fxn
        nmod_t mod
        ulong n_limbs[3]
        ulong ninv_limbs[3]
        void * ninv_huge

    ctypedef fmpz_mod_ctx_struct fmpz_mod_ctx_t[1]

    ctypedef struct fmpz_mod_poly_struct:
        fmpz_struct * coeffs
        slong alloc
        slong length

    ctypedef fmpz_mod_poly_struct fmpz_mod_poly_t[1]

cdef extern from "flint/fmpz_mod.h":
    void fmpz_mod_ctx_init(fmpz_mod_ctx_t ctx, const fmpz_t n)
    void fmpz_mod_ctx_clear(fmpz_mod_ctx_t ctx)

cdef extern from "flint/fmpz_mod_poly.h":
    ctypedef struct fmpz_mod_berlekamp_massey_struct:
        slong npoints
        fmpz_mod_poly_t R0
        fmpz_mod_poly_t R1
        fmpz_mod_poly_t V0
        fmpz_mod_poly_t V1
        fmpz_mod_poly_t qt
        fmpz_mod_poly_t rt
        fmpz_mod_poly_t points

    ctypedef fmpz_mod_berlekamp_massey_struct fmpz_mod_berlekamp_massey_t[1]

    slong fmpz_mod_poly_degree(const fmpz_mod_poly_t poly, const fmpz_mod_ctx_t ctx)
    void fmpz_mod_poly_get_coeff_fmpz(fmpz_t x, const fmpz_mod_poly_t poly, slong n, const fmpz_mod_ctx_t ctx)

    void fmpz_mod_berlekamp_massey_init(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)
    void fmpz_mod_berlekamp_massey_clear(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)
    void fmpz_mod_berlekamp_massey_add_point(fmpz_mod_berlekamp_massey_t B, const fmpz_t a, const fmpz_mod_ctx_t ctx)
    int fmpz_mod_berlekamp_massey_reduce(fmpz_mod_berlekamp_massey_t B, const fmpz_mod_ctx_t ctx)
    const fmpz_mod_poly_struct * fmpz_mod_berlekamp_massey_V_poly(const fmpz_mod_berlekamp_massey_t B)


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
    void nmod_berlekamp_massey_add_point(nmod_berlekamp_massey_t B, ulong a)
    int nmod_berlekamp_massey_reduce(nmod_berlekamp_massey_t B)
    const nmod_poly_struct * nmod_berlekamp_massey_V_poly(const nmod_berlekamp_massey_t B)
    const nmod_poly_struct * nmod_berlekamp_massey_R_poly(const nmod_berlekamp_massey_t B)

cdef extern from "flint/ulong_extras.h":
    ulong n_sqrtmod(ulong a, ulong p)

cdef extern from "flint/qfb.h":
    ctypedef struct qfb_struct:
        fmpz_t a
        fmpz_t b
        fmpz_t c

    ctypedef qfb_struct qfb_t[1]

    void qfb_init(qfb_t q)
    void qfb_clear(qfb_t q)
    int qfb_equal(qfb_t a, qfb_t b)
    void qfb_discriminant(fmpz_t D, qfb_t f)
    void qfb_reduce(qfb_t r, qfb_t f, fmpz_t D)
    void qfb_nucomp(qfb_t r, const qfb_t f, const qfb_t g, fmpz_t D, fmpz_t L)
    void qfb_pow_ui(qfb_t r, qfb_t f, fmpz_t D, ulong exp)
    void qfb_pow(qfb_t r, qfb_t f, fmpz_t D, fmpz_t exp)
    void qfb_inverse(qfb_t r, qfb_t f)
    void qfb_principal_form(qfb_t f, fmpz_t D)
    int qfb_is_primitive(qfb_t f)
    void qfb_prime_form(qfb_t r, fmpz_t D, fmpz_t p)

def berlekamp_massey(seq: list[int], p: int) -> list[int]:
    assert all(x < 2**64 for x in seq)
    cdef nmod_berlekamp_massey_t B
    cdef ulong pi = p
    cdef ulong xi
    nmod_berlekamp_massey_init(B, pi)
    for x in seq:
        xi = x
        nmod_berlekamp_massey_add_point(B, xi)
    nmod_berlekamp_massey_reduce(B)

    cdef const nmod_poly_struct *v = nmod_berlekamp_massey_V_poly(B)
    cdef slong deg = nmod_poly_degree(v)
    res = []
    for i from 0 <= i <= deg:
        res.append(nmod_poly_get_coeff_ui(v, i))
    nmod_berlekamp_massey_clear(B)
    return res

def berlekamp_massey_big(seq: list[int], p: int) -> list[int]:
    pz = flint.fmpz(p)

    cdef fmpz_mod_ctx_t ctx;
    fmpz_mod_ctx_init(ctx, (<fmpz>pz).val);

    cdef fmpz_mod_berlekamp_massey_t B
    cdef fmpz xi
    fmpz_mod_berlekamp_massey_init(B, ctx)
    for x in seq:
        xi = <fmpz>flint.fmpz(x)
        fmpz_mod_berlekamp_massey_add_point(B, xi.val, ctx)
    fmpz_mod_berlekamp_massey_reduce(B, ctx)

    cdef const fmpz_mod_poly_struct *v = fmpz_mod_berlekamp_massey_V_poly(B)
    cdef slong deg = fmpz_mod_poly_degree(v, ctx)
    res = []
    # Extract coefficients by converting to string
    cdef fmpz_t c
    cdef char *cstr
    fmpz_init(c)
    for i from 0 <= i <= deg:
        fmpz_mod_poly_get_coeff_fmpz(c, v, i, ctx)
        cstr = fmpz_get_str(NULL, 16, c)
        a = int((<bytes>cstr).decode(), 16)
        free(cstr)
        res.append(a)
    fmpz_mod_berlekamp_massey_clear(B, ctx)
    fmpz_mod_ctx_clear(ctx)
    return res

def sqrtmod(a: int, p: int):
    assert p < 2**64
    a = a % p
    r = n_sqrtmod(a, p)
    if a != 0 and r == 0:
        raise ValueError("no square root")
    return r

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

    def __eq__(self, other):
        if self is other:
            return True

        if isinstance(other, qfb):
            return bool(qfb_equal(self.val, (<qfb>other).val))

        return False

    def q(self):
        cdef char *cstr
        cstr = fmpz_get_str(NULL, 16, self.val.a)
        a = int((<bytes>cstr).decode(), 16)
        free(cstr)

        cstr = fmpz_get_str(NULL, 16, self.val.b)
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
