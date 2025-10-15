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

A single high-end consumer GPU can compute the class group for
a 512-bit discriminant in a few weeks.

Distributed computation is not supported but appropriate patches can make
running instances of the program on multiple hosts useful.

## Hardware requirements

Pyroclastic is written for cheap, modern consumer GPU. Datacenter-oriented
devices may also be supported if a Vulkan driver is available.

GPU specifications vary wildly and parameters may not be optimized for all
hardware. Manual tweaks are expected to obtain best performance.

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

Multiple GPUs are supported during the `sieve` and `linalg` steps.
The current implementation assumes that all available GPUs are identical
and are numbered `0..N` in the Vulkan API.

The sieve has 2 implementations:

* implementation 1 is a standard SIQS where interval size grows with the factor base size,
  and each polynomial is sieved using roots of all primes from the factor base
  (computationally intensive but low pressure on memory I/O)
* implementation 2 uses the idea from Kleinjung in "Quadratic Sieving"
  to precompute which primes are necessary for each polynomial, which allows
  shrinking interval size to a very small value (less than 1024).
  This requires read/writes to large memory buffer, making the sieve much slower.
  It can be faster (depending on hardware) for discriminant above 400 bits

Cofactorization is handled using CPU only, using the FLINT library, which may create
a bottleneck on hosts with a small CPU and a large GPU. If the Python bindings to
`yamaquasi` (module `pymqs`) is found in the Python environment, it will be used
instead: it usually results in lower CPU usage.

The sparse matrix-vector product has multiple implementations:

* a "naïve" kernel with multiple variants (single modulus, big modulus, multiple moduli),
  which is used for the group structure, and can also be used for the class number computation.
  It is more efficient on NVIDIA devices.
* a "block COO" kernel dividing the matrix in stripes, where sums are accumulated
  in shared memory (instead of registers). It can be used during the class number computation.
  It is more efficient on AMD devices.
* other kernels exist in the repository for benchmarking purposes

## Integer factorization

The quadratic sieve can also be used to factorize integers.
The (work in progress) command `pyroclastic-factor` implements this process.

On a AMD Radeon 780M integrated GPU, RSA-110 can be factorized in about 45 minutes,
and RSA-120 can be factorized in about 4 hours.

On a dual Nvidia RTX 3080 computer, RSA-120 can be factorized in about 45 minutes,
and RSA-130 can be factorized in about 5 hours.

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

