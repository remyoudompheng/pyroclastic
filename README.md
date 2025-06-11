# Pyroclastic: computation of quadratic class groups

Pyroclastic is a small implementation of the quadratic sieve
to compute the structure of class groups for **imaginary
quadratic fields**. It relies on the following components:

* [Vulkan](https://docs.vulkan.org/) for GPU compute shaders
* [Kompute](https://github.com/KomputeProject/kompute) for Python bindings
* [python-flint](https://python-flint.readthedocs.io) for computer algebra

It is written in Python to allow easy inspection of internal computations.

Building the project requires Cython for bindings to several FLINT
APIs which are not available in python-flint 0.7.

The current implementation should be suitable for discriminants
between 150 and 550 bits. Larger inputs may be supported depending
on hardware, smaller inputs may be slower than equivalent CPU implementations.

## Hardware requirements

Pyroclastic is written for cheap, modern consumer GPU. GPU specifications
vary wildly and parameters may not be optimized for all hardware.

The quadratic sieve steps assumes GPU with fast _local memory_ and
works best on architectures with more than 64kB local memory per core.
On modern GPU architectures, local memory will usually allow 16 or 32
atomic additions per clock cycle, whereas a modern CPU usually allows only
1-3 L1 random writes per cycle.

For example, when using a Ryzen 7840HS APU, the integrated GPU (Radeon 780M
with 12CU) will sieve 4x faster than the 8 CPU cores using `yamaquasi` with
similar parameters.

## Computation steps

The class group computation consists of the following steps:

1. Quadratic sieve (SIQS variant) to generate relations in the class group
2. Class number computation, which computes the index of the relation lattice
   through determinants modulo small primes
3. Class group structure, which computes matrix kernels modulo prime factors
   of the class number

Each step is available as a separate command:

* `pyroclastic-sieve`: performs the sieve and creates the initial relation file
* `pyroclastic-relations`: eliminates relations to build a smaller matrix
* `pyroclastic-linalg`: computes the class number using large determinants
* `pyroclastic-groupstruct`: computes an explicit isomorphism with a product of cyclic groups

An additional command `pyroclastic-smoothrel` can be used to compute
coordinates for an arbitrary prime ideal. It can run quickly even with
a low-power GPU.

## Bibliography

Henri Cohen, Calcul du nombre de classes d'un corps quadratique imaginaire ou
réel, d'après Shanks, Williams, McCurley, A. K. Lenstra et Schnorr
Séminaire de théorie des nombres de Bordeaux, Série 2, Tome 1 (1989) no. 1, pp. 117-135.
<http://www.numdam.org/item/JTNB_1989__1_1_117_0/>

Michael Jacobson, Applying sieving to the computation of class groups
Math. Comp. 68 (226), 1999, 859-867
<https://www.ams.org/journals/mcom/1999-68-226/S0025-5718-99-01003-0/S0025-5718-99-01003-0.pdf>

Jean-François Biasse, Improvements to the computation
of ideal class groups of imaginary quadratic number fields
<https://arxiv.org/pdf/1204.1300.pdf>

Thorsten Kleinjung, Quadratic Sieving
Math. Comp. 85 (300), 2016, 1861-1873
<https://www.ams.org/journals/mcom/2016-85-300/S0025-5718-2015-03058-0/S0025-5718-2015-03058-0.pdf>

Victor Pan, Computing the determinant and the characteristic polynomial of a matrix via solving linear systems of equations,
Information Processing Letters, Volume 28, Issue 2, 1988
<https://doi.org/10.1016/0020-0190(88)90166-4>

Erich Kaltofen, Gilles Villard. On the complexity of computing determinants
LIP RR-2003-36, Laboratoire de l’informatique du parallélisme. 2003, 2+35p. hal-02102099
<https://hal-lara.archives-ouvertes.fr/hal-02102099/file/RR2003-36.pdf>

