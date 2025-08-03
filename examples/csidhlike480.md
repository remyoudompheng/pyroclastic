# CSIDH-like 480-bit prime

In this computation, the discriminant is:
```
D = 1 - 4 p_1 ... p_k
```
where `p_i` are primes from the list
```
3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 53, 59, 61, 67, 71, 73, 79,
83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
359, 367
```

The number -D is a 480-bit prime with 145 digits
```
|D| = 3099560651622315083474139948405667430611730311991839527640502560591039101776719929938238527027620424244726453640516135283387761031653922361710339
```

The class number formula can be used to obtain a rough approximation:
```
h ~= 1.925e+72
```

The actual class number is

```
h = 1925145440260356888328016187294556389335064523883577129459100294843766331
```

Implementation was based on commit 179dfc436c73a6d238b219d030913eee25d4d465.

## Sieving parameters

This computation was done without the SIQS initialization optimization (similarly to the 460-bit computation)

* the sieving bound was B1=4000000 (141783 primes)
* the bound for large primes was 60 B1 = 240M
* polynomials had leading coefficients A with 12 factors (2048 polynomials each)
* each polynomial was sieved over an interval of size 589824 (36 segments of 16kB)
* the threshold for reporting sieve results was 175 bits (3 large primes allowed)

The sieving step ran on a computer with 4 AMD Radeon Pro VII GPUs
with a dual Intel Xeon Silver 4108 CPU, using Vulkan compute shaders
through the open-source RADV driver in [Mesa](https://mesa3d.org/).

* The average sieving speed was 200G/s (166 A values per second)
* Relations were found once every 31.4 billion elements (6.4 relations per second)
* 2582273 relations were found involving 2880893 primes
* 26221 relations have no large prime
* 439929 relations have 3 large primes
* The process took 404737 seconds and sieved a total region of size `8.1 × 10^16`.
* The cost of cloud resources was 6.18 USD (112.4 hours)

## Partial elimination of relations

This step has a negligible cost and was run on an ordinary computer.

The resulting matrix has 176771 columns, 177014 rows (relations)
with an average number of 81 coefficients per row (largest coefficient is 25).

A duplicate relation had to be manually removed to proceed to the next step.

## Computation of class number

This computation was done using the ordinary Wiedemann algorithm on multiple
word-sized moduli simultaneously. This step ran on a computer with a dual
Intel Xeon Silver 4314 CPU and a single Nvidia RTX 4090 GPU. Multiple CPU threads (6)
were used to maximize GPU occupancy. The implementation used the standard
Vulkan driver from NVIDIA.

* Size of moduli for small prime determinants was 57 bits
* The computation kernel was `spmv_multi.comp` with 32 moduli and 16 rows per workgroup
* The average throughput was about 41800 matrix-vector multiplications per second
* First determinant had size 222032 bits and was computed in 33590s (using 3968 moduli)
* Second determinant has size 222021 bits and was computed in 33532s (using 3968 moduli)
* The GCD of determinants was 32h=61604654088331420426496517993425804458722064764274468142691209435000522592
* The cost of cloud resources was 3 USD (18.6 GPU-hours)

## Determination of group structure

This step is asymptotically negligible and runs on an ordinary computer
in a short time.

For a basis of 176771 primes, the structure modulo small factors of `h`
(3^2, 1733549, 2609417, 56406101) is done using a linear complexity
generic hash table search (with a bit of BSGS), and takes 150 seconds
for each factor on a single CPU thread.

The structure modulo the largest factor 838330652033520585208664041776878989517738440355323
is obtained by linear algebra (900s of GPU time on Radeon Pro VII), as the right kernel
of the relation matrix.

```
Invariants [9, 1733549, 2609417, 56406101, 838330652033520585208664041776878989517738440355323]
3 [1, 1, 1, 1, 1]
5 [8, 1311663, 1959165, 38681018, 52237678800729892682984727563955114557462053154049]
7 [0, 1194638, 1157605, 24909885, 650668665998032064829783051307463435345720022710554]
11 [5, 309700, 1821009, 22650393, 371534566029753324463747573154263408874466050007399]
...
1000931 [1, 1503410, 1750773, 55300433, 334869004822865237391863081676759244077904569700429]
1000969 [5, 1721612, 865138, 3038810, 85386372827913876457367171745097264127512091792007]
1000999 [4, 102827, 1557609, 44041250, 9381438909742459914020467073622399603575116188914]
1001023 [3, 326631, 1178477, 53390148, 709760972684009383522959772620944682136605836077316]
1001027 [8, 406990, 533652, 19677865, 571575837541801296326530196680518971066739168632511]
```

The class group is cyclic, and `[3]` is a group generator. The following identity is true:
```
[3]^1610907629621262463080865223326947897029550743850643309000378872245447095 = [1000999]
```
