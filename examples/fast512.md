# Fast computation of a class group with 256-bit order

The goal of this computation is to select a discriminant such
that $\rm{Cl}(\mathbb Q[\sqrt{-p}])$ is a group of order close to $2^{256}$,
but such that the computation is as cheap as possible.

The entire computation took 22.2 days using a single Nvidia RTX 5080 GPU.

## Selection of an optimal discriminant

By the class number formula and the usual heuristics, best candidates are
discriminants which are squares modulo many small primes: they have the
property that the quadratic sieve provides relations with a higher
density, and also that the class number has the largest ratio to $\sqrt{|D|}$.

Quick numerical experiments show that the largest ratio $h / \sqrt(D)$ we can obtain
is slightly larger than 3.7, so we focus on primes which are close to $2^{512} / 3.7^2$.

Since all criteria are approximate, we can add the constraint that $D \mod l = 1$
for many small primes (which is inspired by isogeny research).

The problem is then similar to the _pseudosquare_ problem, or the problem
of integers which are "locally squares" (see for example Daniel J. Bernstein,
(_Doubly-focused enumeration of locally square polynomial values_)[https://cr.yp.to/focus/focus-20030928.pdf] )

We will try to optimize the following quantity (where $(D|p)$ is 2 if p=2 and D=8k+1):

$$ \alpha(D) = \sum_p (D|p) \log_2 p / p $$

In practice, there is a significant correlation ($h / \sqrt(D) \simeq 0.34 \alpha + 0.93$).

In the CSI-Fish computation, we have $\alpha(D) \simeq 6.50$ and $h(D) \simeq 1.163 \sqrt{|D|}$.
In the csidhlike512 computation, we have $\alpha(D) \simeq 5.77$ and $h(D) \simeq 1.078 \sqrt{|D|}$.

Following djb, we can proceed as following using a CRT approach:

* let $P_1$ be the product of 8 and off primes from 3 to 300
* let $P_2$ be the product of primes between 300 and 350
* let $P_3$ be the product of primes between 350 and 390

The product $P_1 P_2 P_3$ is larger than $2^{512}$.

We will consider primes of the form:

$$ p = P_1(P_2 P_3 - U P_2 - V P_3) - 1 $$

such that $p$ is a square modulo $P_2$ and $P_3$. The values of $U$ and $V$
are brute-forced to obtain numbers close to $2^{512} / 3.7^2$ and the values
$D = -p$ giving the highest values of $\alpha$ are examined.

By construction, if $U$ and $V$ are selected so that $-p$ is a quadratic residue
modulo $P_1 P_2 P_3$, a positive contribution to $\alpha(-p)$ is guaranteed,
and $p$ is not divisible by primes $\leq 380$ which increases the probability
to be a prime number (about 3%).

In practice, we can obtain $\alpha > 8.2$ corresponding to class
numbers $h > 3.7 \sqrt{|D|}$.

After spending several core-hours to process a few million primes, the
following candidates with $\alpha > 8.24$ are found (about 1 per million):

```
p=0x127213389856258bd2c6085912010f78fbbed408d918652adbe51295ba32de54e478da39e837e80d5e488d8ac3f889d8cb9469e49dde8d70192a6ed1a56c33ff
p=0x1224d992a337359c6fbe0c8897617071d80264821531704a4c40b04c4c62b1a84bc7301fa2fa365ec3bf14387dcbbb064fa60d0177d021c8379c52ec93688087
p=0x122cd3ce525f3f60f9cd32ea3f0827d72cec419c609c0e16786d0a76628c00f710fc3f9cd1be9b91befd1abaa195de64d9e5881b65fcff42a275eb930476bfaf
p=0x127a90dfd4b9cbfcabcb3dd01e038097b65fc399790c04f4bb5a0f940322283d82e70d3aefa2503610cf64d40ee99cc3c9b3739eb8865ec6ad8adafc66ef100f
```

The last one is selected (α=8.36), its class number is predicted to be about `1.02 × 2^256`.

Numerical experiments show that:

* with identical sieve parameters, the number of relations is 2.5x larger than `csidhlike512`
* with identical sieve parameters, the number of relations is 5x larger than `pegasis509`

## Selected discriminant

The discriminant chosen for the computation is:

```
D = -p = -967811877341830792750463701695460944681653670520001666865429369224203090000819223881442353184785953753633825775038816840982464805933203249883125864402959
```

It has the property that:
```
p + 1 = 2^4 * 3 * 5 * 7 * 11 * 13 * 17 * 19^2 * 23 * 29 * 31^2 * 37^2 * 41 * 43
* 47 * 53 * 59 * 61 * 67 * 71 * 73 * 79 * 83 * 89 * 97 * 101 * 103 * 107 * 109
* 113 * 127 * 131 * 137 * 139 * 149 * 151 * 157 * 163 * 167 * 173 * 179 * 181
* 191 * 193 * 197 * 199 * 211 * 223 * 227 * 229 * 233 * 239 * 241 * 251 * 257
* 263 * 269 * 271 * 277 * 281 * 283 * 293 * 691 * 48121 * 81684030176557631639
```

We observe that D is a square modulo 8 and modulo all 99 odd primes <= 541
except 431, 467, 491, 509. Looking at all odd primes less than 5000,
D is a square modulo 2/3 of these primes.

The class number formula gives the approximation:

$$ h \simeq 1.1811 × 10^{77} \simeq 1.020 × 2^{256} $$

## Sieving step

This step uses the improved initialization method (as in [pegasis509](./pegasis509.md) ),
enabled by option `--siever2`. For identical sieving bounds, this improves
the speed of relation collection 3-fold.

To take into account the higher frequency of relations, sieving bounds are lowered:

* the sieving bound was B1=6000000 (207079 primes)
* the bound for large primes was 80 B1 = 640M
* polynomials had leading coefficients A with 21 factors (1048576 polynomials each)
* each polynomial was sieved over an interval of size 512 (32768 segments of 8kB)
* the threshold for reporting sieve results was 162 bits

The sieve was run on a single Nvidia RTX 5080 GPU supported by an (old)
Intel Xeon E5-2650 v4 CPU.

* Sieving speed was 18.4G/s with 6.11 relations per second
* Sieving duration was 13 days (6875003 relations, region size 20.7 × 10^15)
* 13163 relations have no large prime
* 220561 relations have 1 large prime
* 1547255 relations have 2 large primes
* 5089906 relations have 3 large primes
* 4118 relations have 4 large primes

## Computation of class number

Structured Gauss elimination reduces the matrix to 329277 columns and 329518 rows.

Linear algebra is done with shader `spmv_multi` with batches of 32 moduli
using 55-bit primes, and block Wiedemann parameter `BLOCKM=2`.

* Each batch of 32 moduli (size 55 bits) took 1300s GPU time (on average)
* Processing speed was equivalent to 11540 matrix-vector multiplications per second
* The first determinant has size 504559 bits and was computed using 9344 moduli in 398011s
* The second determinant has size 504561 bits and was computed using 9344 moduli in 399898s
* Total processing time (on the same RTX 5080 device) is 9.2 days

The computed class number is:

```
h = 118108182256781374775833425644860844013221204294503800379122602915672006380867
h = 0x1051edcb8bbd01f60a634684e67be965675e86a284c19ba66d55a5cbacffccd43
h = 13 * 112749223 * 2405197467437 * 1101097822225426426810351 * 30426110208467456847913800579059
```
