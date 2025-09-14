# PEGASIS 509-bit prime

This computation considers the smallest prime from publication

Pierrick Dartois, Jonathan Komada Eriksen, Tako Boris Fouotsa, Arthur Herlédan Le Merdy, Riccardo Invernizzi,
Damien Robert, Ryan Rueger, Frederik Vercauteren, Benjamin Wesolowski.
PEGASIS: Practical Effective Class Group Action using 4-Dimensional Isogenies.
https://eprint.iacr.org/2025/401

The discriminant is the (negated) 153-digit prime number:

```
D = 1 - 33 × 2^503
D = -864175120484581453683482079962486176185193500155369104423588921177379322250834082489183304374038697487834084609675858746433355728113743766078731283595263
```

This number is smaller than the one considered in Csi-Fish: however
it is such that the quadratic sieve will have a noticeably lower yield,
making the computation more expensive.

The class number formula can be used to obtain a rough approximation:
```
h ~= 3.0118e+76
```

The actual class number is

```
h = 30115877202646647376923175490783656066559648055235996072894708571193049262827
```

It was computed in August-September 2025 in about 36 days using cloud GPU
resources. The implementation was a minor variation of commit
d929d6bb186597a2b523658bae154a82bd1431d9 with manually selected parameters.

## Sieving parameters

This computation used the method described by Kleinjung in
[Quadratic Sieving](https://www.ams.org/journals/mcom/2016-85-300/S0025-5718-2015-03058-0/S0025-5718-2015-03058-0.pdf).

In `pyroclastic` GPU implementation it makes the sieving speed
much lower, but produces relations at a higher rate for large
discriminants (above 130 decimal digits).

* the sieving bound was B1=8000000 (269286 primes)
* the bound for large primes was 80 B1 = 640M
* polynomials had leading coefficients A with 21 factors (1048576 polynomials each)
* each polynomial was sieved over an interval of size 512 (32768 segments of 8kB)
* the threshold for reporting sieve results was 162 bits

Quick benchmarks show that using the older "traditional" method
with 2048 polynomials per A coefficients and intervals of size 1M
produced relations 2x slower on a Nvidia RTX 5070 GPU.

Due to technical issues, relations were produced in multiple runs:

* Two interrupted runs produced 272837 and 294212 relations
* A 4x Nvidia RTX 5070 machine ran for 175 hours and produced 2695182 relations
  (sieve region size 24 × 10^15 at speed 38.8G/s, 4.27 relations per second)
* A 2x Nvidia RTX 5080 machine ran for 486 hours and produced 6494544 relations
  (sieve region size 58.6 × 10^15 at speed 33.5G/s, 3.72 relations per second)
* Due to overlap between random generator outputs, 511206 relations were duplicated
* 14215 relations have no large prime
* 260439 relations have 1 large prime
* 1969135 relations have 2 large primes
* 6994934 relations have 3 large primes
* 6836 relations have 4 large primes

## Structured Gauss elimination

This step has a negligible cost and takes about 20 minutes on a single CPU core.

The elimination depth is chosen to produce a matrix with 511199 rows
and 510900 columns (after elimination of residual redundant relations
using the `matrepair` tool).

This matrix has an average of 116 coefficients per row (at most 240).

## Computation of class number

Compared to the [previous computation](./csidhlike512.md) several improvements
were made:

* a block Wiedemann approach is used for the determinant, with parameter m=3
  (3 linear sequences produced simultaneously), reducing required matrix
  multiplications by 33% (but with drastically higher CPU cost)
* matrix rows are now sorted to reduce divergence inside workgroups
* the GPU kernel was modified so that each invocation handles 4 moduli at a time
  to attempt more aggressive memory fetches

The computation was run on a computer with 2x NVIDIA RTX 5080 GPU and
a 32-core EPYC 7532 (Zen 4) CPU.

* The average CPU usage was around 40% (a parameter m=2 could be used
  to reduce CPU usage)
* Each batch of 32 moduli (size 55 bits) took 1850s GPU time (on average)
  and around 600s on 32 CPU cores.
* The matrix multiplication kernel takes about 2.6ms and the total equivalent
  throughput was about 24000 matrix multiplications (for 1 modulus) per second
  across the 2 GPUs
* The average time for computing 1 small determinant was 29 seconds
  (124 moduli per hour)

The first determinant was computed in 106 hours using determinants
modulo 12320 small primes. It is a 675728-bit number (203415 decimal digits).

By trial factoring, P-1 or ECM algorithm, it is possible to find small factors
of the class number: 101, 409, 3943, 17851, 5503889543116441.
The factor 5503889543116441 can be found using GMP-ECM with B1=3000 after 30 curves,
which takes a few minutes on a small computer.

The following binary forms have small prime order: they are computed by raising
a small prime form to the power `det / l` for each small factor.
```
order 101
Qfb(923330238552856320028732590096262368957949447996561625333480363496766714234,
    789291323327107443205762368981205914166642875565143081122435107728594513057,
    234151895272342280193397544017112837373689601521307269300195665889453870625992)

order 409
Qfb(2282562858511068842530337906699259957680565799584739571485360568923077296083,
    1578106155695478176189095177434650302099244105527810513124440136839022488515,
    94922417611815211409808094479835233901201740548550409390288804537597492650134)

order 3943
Qfb(6599163698752622586966260326340396348298231294016376728113308797565153075534,
   -1926596310969227885686938195908640340426184215458386886172134875457343136007,
    32878669686360035714453449223816026858638827985601617745727517651827906173642)

order 17851
Qfb(3022243494616964850426063207161653448848473671395750380550688646389839709948,
    3007163415279307957270741595351598444684133893252668639991626786117467685687,
    72232610794439503026282664991875359979313446697446704563972225958515367833571)

order 5503889543116441
Qfb(10126251657497659126004907204439469217728639606637108789706841625847915094267,
    5279555815581039461251055889132082055321582726267138092154476165629278145711,
    22023174523663156411825823182144391029947471829416951071585832709287256101688)
```

The second determinant was computed in 100 hours using determinants
modulo 12288 primes (the script stops after 12320 primes).
It is a 675729-bit number (203415 decimal digits).

The GCD of both determinants is equal to 4h=120463508810586589507692701963134624266238592220943984291578834284772197051308.

This gives the last factor of the class number:
1881879447649265588797081861911138024133873569131.
