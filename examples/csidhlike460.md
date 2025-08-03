# CSIDH-like 460-bit prime

In this computation, the discriminant is:
```
D = 1 - 4 p_1 ... p_k
```
where `p_i` are primes from the list
```
3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 233, 239, 241, 251,
257, 263, 269, 271, 277, 281, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353
```

The number -D is a 460-bit prime with 139 digits
```
|D| = 2849259712428984621020777351719488569102012732818373208098676861865327838960598631837611153460707778466649073855794448192313020119118490459
```

The class number formula can be used to obtain a rough approximation:
```
h ~= 1.865e+69
```

The actual class number is

```
h = 1865152088992494287216620839733414356505938142231059832935934817557943
```

Implementation was based on commit 61364de6e3fd8d0c53483bfd234cfcc6cebcff2d.

## Sieving parameters

This computation was done without the SIQS initialization optimization
from [Kleinjung, Quadratic Sieving](https://www.ams.org/journals/mcom/2016-85-300/S0025-5718-2015-03058-0/S0025-5718-2015-03058-0.pdf)

* the sieving bound was B1=3000000 (109110 primes)
* the bound for large primes was 55 B1 = 165M
* polynomials had leading coefficients A with 12 factors (2048 polynomials each)
* each polynomial was sieved over an interval of size 491520 (30 segments of 16kB)
* the threshold for reporting sieve results was 169 bits (3 large primes allowed)

The sieving step ran on a computer with 4 AMD Radeon Pro VII GPUs
with a dual Intel Xeon Silver 4108 CPU, using Vulkan compute shaders
through the open-source RADV driver in [Mesa](https://mesa3d.org/).

* The average sieving speed was 193G/s (192 A values per second)
* Relations were found once every 15.8 billion elements (12.2 relations per second)
* 1907478 relations were found involving 2129842 primes
* 13861 relations have no large prime
* 226344 relations have 1 large prime
* 1526792 relations have 2 large primes
* 140481 relations have 3 large primes
* The process took 156150 seconds and sieved a total region of size `3.01 × 10^16`.
* The cost of cloud resources was 2.40 USD (43.4 hours)

## Partial elimination of relations

This step has a negligible cost and was run on an ordinary computer.

The resulting matrix has 115452 columns, 115735 rows (relations)
with an average number of 79 coefficients per row (largest coefficient is 21).

## Computation of class number

This computation was done using the ordinary Wiedemann algorithm on multiple
word-sized moduli simultaneously. This step ran on a computer with a dual
Intel Xeon E5-2650 v4 CPU and a single Nvidia RTX 5080 GPU. Multiple CPU threads (8)
were used to maximize GPU occupancy. The implementation used the standard
Vulkan driver from NVIDIA.

* Size of moduli for small prime determinants was 57 bits
* The computation kernel was `spmv_multi.comp` with 32 moduli and 16 rows per workgroup
* The average throughput was about 45000 matrix-vector multiplications per second
* First determinant had size 147183 bits and was computed in 13701s (using 2656 moduli)
* Second determinant has size 147171 bits and was computed in 13723s (using 2656 moduli)
* The GCD of determinants was 2h=3730304177984988574433241679466828713011876284462119665871869635115886
* The cost of cloud resources was 0.65 USD (7.6 GPU-hours)

## Determination of group structure

This step is asymptotically negligible and runs on an ordinary computer
in a short time.

For a basis of 110883 primes, the structure modulo small factors of `h`
(37, 107, 127, 1049, 102587, 355763) is done using a linear complexity
generic hash table search, and takes 100 seconds for each factor
on a single CPU thread.

The structure modulo the largest factor 96893898884306949897748381723923531108443372609879
is obtained by linear algebra (1000s of GPU time on Radeon 780M), as the right kernel
of the relation matrix.

```
Invariants [37, 107, 127, 1049, 102587, 355763, 96893898884306949897748381723923531108443372609879]
3 [1, 0, 1, 1, 1, 1, 1]
5 [4, 1, 45, 687, 62606, 94958, 92623762490980848777541114823106981266111407735605]
7 [12, 63, 119, 583, 41394, 258606, 53052022781812873100816844967342481419524494957754]
11 [34, 89, 101, 866, 8910, 224849, 36832998826504307862396551262069274792046731063867]
...
65449 [26, 58, 85, 481, 76107, 205988, 89276472233722894405063357392001868805548552509086]
65497 [32, 84, 59, 177, 35536, 195284, 82938096259853545365483248850811407596072171272766]
65519 [12, 99, 60, 911, 90911, 69029, 55245488239168186974086593720025889235539708761003]
65537 [22, 80, 26, 514, 49076, 120055, 86502074393924325737191182901851271604142718485975]
65543 [0, 73, 21, 503, 35657, 141137, 85926056573918637868373037192295447470795661483623]
65551 [34, 8, 58, 2, 20425, 304355, 36430502157918821021988807414622848876999080427575]
65579 [30, 56, 26, 184, 75786, 19464, 90582238886375397159762444475926384523069566112753]
65599 [3, 31, 30, 22, 38827, 317055, 41533802664652229248991024851659858096569621943346]
```

In particular, ideal `[5]` is a group generator, and the following identity is true:
```
[5]^660455195668488043459173795229534754776747377402809931004088203411405 = [65537]
```

Using all relations, discrete logarithms for 1501193 primes can be computed.

## Computation of a class group discrete logarithm

Using the previous computation, discrete logarithms can be obtained for arbitrary
prime ideals.

For example:
```
# next_prime(10**99)
l = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000289

# after quick CPU sieve
[l] = 17^-1 883^1 1361^-1 1439^1 4931^-1 12713^-1 17909^1 28751^-1 499687^-1 85237^-1 105533^1 245131^1 3224779644109117^-1 11306206563739310207^1

# after GPU sieve
3224779644109117 = [2711, -347, 3, 11, 11, -29, -131, -179, 3739, 9439, 41507, -92431, 109517, -382813, -854333, 884999, 1547437, -4294723, 38984107,
  22993, 23021, 23081, 23173, -23189, -23209, 23251, -23269, 23321, 23473, 23599]

# after GPU sieve
11306206563739310207 = [3, 47, -89, 1601, -1109, -13003, -19289, -62401, -116273, 126421, 152459, 184649, -308923, -781171, -883721, 1763011, 3827401,
  -10733, -10799, -10909, -10993, 11047, -11059, 11113, -11159, 11213, -11273, 11411]

# final decomposition
[l] = 11^-2 17^-1 29^1 47^1 89^-1 131^1 179^1 347^1 883^1 1109^-1 1361^-1
1439^1 1601^1 2711^-1 3739^-1 4931^-1 9439^-1 10733^-1 10799^-1 10909^-1
10993^-1 11047^1 11059^-1 11113^1 11159^-1 11213^1 11273^-1 11411^1 12713^-1
13003^-1 17909^1 19289^-1 22993^-1 23021^-1 23081^-1 23173^-1 23189^1 23209^1
23251^-1 23269^1 23321^-1 23473^-1 23599^-1 28751^-1 41507^-1 62401^-1 85237^-1
92431^1 105533^1 109517^-1 116273^-1 126421^1 152459^1 184649^1 245131^1
308923^-1 382813^1 499687^-1 781171^-1 854333^1 883721^-1 884999^-1 1547437^-1
1763011^1 3827401^1 4294723^1 38984107^-1
```

These relations are obtained in a few seconds on a cheap computer.
Using these relations, the following identity can be verified:

```
[l] = [5]^551251910229567266444686187423669701188831587652860406933430607002648
```
