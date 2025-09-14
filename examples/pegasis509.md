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
* The relations involve 10.4 million primes

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

## Determination of group structure

This step was run on the same host as linear algebra: implementation
is not parallel and used a single CPU core and a single GPU.
It also does not use block Wiedemann for large factors.

It took:

* 1.3 hours of CPU for coordinates modulo 101, 409, 3943, 17851
* 0.5 hours of 1 GPU for coordinates modulo 5503889543116441
  (with about 900 matrix-vector multiplications per second)
* 0.8 hours of 1 GPU for coordinates modulo 1881879447649265588797081861911138024133873569131
  (with about 600 matrix-vector multiplications per second)

The class group is cyclic and generated by ideal `[2]`.

The coordinates of the smallest prime ideals are:

```
2 1
3 22912493069467962124474063777480763971815779403004995793796325984979959525481
7 25422256145521216333335649830033177678313070386727838967254307257411438418062
11 7203384133178685252449111713302892094743868652231000279098382586213089736845
13 19730585057699583167794997455087509066588437498872092456157303060874804234721
29 20423561172032371382623422297341870235843105456569548469215739436949525997176
31 5912665443220954377545364683383156123290544493592421596919206898430745049704
37 24858871999942806816462084957928105548024686764258294140787357297815903155030
43 23980927479335672831899714349905044677958096664227682798849525164913785226365
53 2435922845746058717606888121233507799419267209810241976964058489890527554800
...
10000019 22628402470148172399774323745863744157084738124717437721431678835753259962308
10000121 19970517351916212680874094944038741055724355855960475339949513385876645041676
10000141 5114167477099025042342101656662092118178280438809625350530179719055504042859
10000253 19695344706437756719329643993297201249388543081322886618667662453743881362945
10000453 7391936472847361868886105514652696488237867638923455911305080079437106936844
```

Using additional relations, coordinates can be computed for 6712801 prime ideals.

```
639998857 16372840040271515867282097277722739856875221186730399426304844455727105769833
639999203 2883147038097361357069490894249474988780091062376634859538470629315186546387
639999233 7267002756026810803867114471145978622224377170941383501899782025386854252140
639999691 7296970656944109714541518636735754554290297508450115044641398761916166554625
639999749 21697644889206608220528731122690000646372403770060699152765686186399868550087
```

## Computation of coordinates of an arbitrary ideal

The precomputation allows determination of coordinates for arbitrary prime ideals
in a few minutes.

For example

```
# next_prime(10**99)
l = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000289

# after quick CPU sieve
[l] = 2^-3 3^-1 13^1 53^-1 197^-1 4951^-1 10193^-1 10433^-1 37649^-1 90407^1 216641^1 1430281^-1 1354651^-1 6313889^-1 12587322057791752727^1 7056229257950327489^1

# GPU sieve
[12587322057791752727] =  [2, 2, -4649, -12659, 141263, 324991, 1140431,
-2251979, -5909287, -6775607, -43, 163, 239, -317, 180749, 19292969, -77305421,
-82499839, 46153, -46271, -46309, -46523, 46559, -46639, 46681, -46687, 46723,
-46769, 46811]

# GPU sieve
[7056229257950327489] = [2, 2, 1361, 3, 7, -379, -19571, -32507, -73327, 317353,
902669, 1372867, -3644339, 5232863, -6157097, -22829633, 51650129, 127548079,
-48757, -48781, -48869, -48883, -48953, -49031, 49037, 49123, -49297, 49451,
-49613]

# final decomposition
[l ] = 2^1 7^1 13^1 43^-1 53^-1 163^1 197^-1 239^1 317^-1 379^-1 1361^1 4649^-1
4951^-1 10193^-1 10433^-1 12659^-1 19571^-1 32507^-1 37649^-1 46153^1 46271^-1
46309^-1 46523^-1 46559^1 46639^-1 46681^1 46687^-1 46723^1 46769^-1 46811^1
48757^-1 48781^-1 48869^-1 48883^-1 48953^-1 49031^-1 49037^1 49123^1 49297^-1
49451^1 49613^-1 73327^-1 90407^1 141263^1 180749^1 216641^1 317353^1 324991^1
902669^1 1140431^1 1354651^-1 1372867^1 1430281^-1 2251979^-1 3644339^-1
5232863^1 5909287^-1 6157097^-1 6313889^-1 6775607^-1 19292969^1 22829633^-1
51650129^1 77305421^-1 82499839^-1 127548079^1
```

The following identity is found:
```
sage: q0 = pari.qfbprimeform(D, 10**99+289)**1
sage: q1 = pari.qfbprimeform(D, 2)**28747956045024046590662777042383493715348463247270035842120394658100924627239
sage: q0 == q1
True
```
