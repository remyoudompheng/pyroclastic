# Fast computation of a class group with 192-bit order

The goal of this computation is to select a discriminant such
that $\rm{Cl}(\mathbb Q[\sqrt{-p}])$ is a group of order close to $2^{192}$,
but such that the computation is as cheap as possible.

## Selection of an optimal discriminant

By the class number formula and the usual heuristics, best candidates are
discriminants which are squares modulo many small primes: they have the
property that the quadratic sieve provides relations with a higher
density, and also that the class number has the largest ratio to $\sqrt{|D|}$.

Quick numerical experiments show that the largest ratio $h / \sqrt{D}$ we can obtain
is slightly larger than 3.6, so we focus on primes which are close to $2^{384} / 3.6^2$.

Since all criteria are approximate, we can add the constraint that $D \mod l = 1$
for many small primes (which is inspired by isogeny research).

The problem is then similar to the _pseudosquare_ problem, or the problem
of integers which are "locally squares" (see for example Daniel J. Bernstein,
(_Doubly-focused enumeration of locally square polynomial values_)[https://cr.yp.to/focus/focus-20030928.pdf] )

We will try to optimize the following quantity (where $(D|p)$ is 2 if p=2 and D=8k+1):

$$ \alpha(D) = \sum_p (D|p) \log_2 p / p $$

In practice, there is a significant correlation ($h / \sqrt{D} \simeq 0.39 \alpha + 0.42$).

For reference, in the CSI-Fish computation,
we have $\alpha(D) \simeq 6.50$ and $h(D) \simeq 1.163 \sqrt{|D|}$.

Following djb, we can proceed as following using a CRT approach:

* let $P_1$ be the product of 8 and off primes from 3 to 235
* let $P_2$ be the product of primes between 235 and 265
* let $P_3$ be the product of primes between 265 and 285

The product $P_1 P_2 P_3$ is larger than $2^{384}$.

We will consider primes of the form:

$$ p = P_1(P_2 P_3 - U P_2 - V P_3) - 1 $$

such that $p$ is a square modulo $P_2$ and $P_3$. The values of $U$ and $V$
are brute-forced to obtain numbers close to $2^{384} / 3.6^2$ and the values
$D = -p$ giving the highest values of $\alpha$ are examined.

By construction, if $U$ and $V$ are selected so that $-p$ is a quadratic residue
modulo $P_1 P_2 P_3$, a positive contribution to $\alpha(-p)$ is guaranteed,
and $p$ is not divisible by primes $\leq 285$ which increases the probability
to be a prime number.

In practice, we can obtain $\alpha > 7.8$ corresponding to class
numbers $h > 3.6 \sqrt{|D|}$.

After a few minutes, we can find an interesting candidate:

```
p=0x139c0df93b7fc96971d9a3509b1010a507f1b3286da74633e1a61b35e2620fac97bdf17abeb807332f57fe2d58734b3f
```

This number has α=7.956, and its class number is predicted to be about `1.002 × 2^192`.

## Selected discriminant

The discriminant chosen for the computation is the 381-bit prime:

```
D = -p = -3018191861484713288852891454778967324156875724275015750648806777976749878367066636171055110285343919841589962427199
```

It has the property that:
```
p + 1 = 2^6 * 3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
* 53 * 59 * 61 * 67 * 71 * 73 * 79 * 83 * 89 * 97 * 101 * 103 * 107 * 109 * 113
* 127 * 131 * 137 * 139 * 149 * 151 * 157 * 163 * 167 * 173 * 179 * 181 * 191
* 193 * 197 * 199 * 211 * 223 * 227 * 229 * 233^2 * 11894083 * 1531246589
```

We observe that D is a square modulo 2/3 of all primes less than 5000.
For example, it is a square modulo all primes between 400 and 600 except
6 of them (433, 443, 449, 523, 557, 593).

The class number formula gives the approximation:

$$ h \simeq 6.29 × 10^{57} \simeq 1.002 × 2^{192} $$

## Sieving step

This step uses the improved initialization method (as in [pegasis509](./pegasis509.md) ),
enabled by option `--siever2`.

* the sieving bound was B1=1000000 (39263 primes)
* the bound for large primes was 25 B1 = 25M
* polynomials had leading coefficients A with 19 factors (262144 polynomials each)
* each polynomial was sieved over an interval of size 512 (16384 segments of 8kB)
* the threshold for reporting sieve results was 119 bits

The sieve was run on a Radeon 780M integrated GPU (Ryzen 7840HS).

* Sieving speed was 3.7G/s with 110 relations per second (6 threads)
* The total sieve region size was 15.4 × 10^12
* 460039 relations were collected (70 minutes) involving 499578 primes
* 58425 relations have no large prime
* 56284 relations have 1 large prime
* 225497 relations have 2 large primes
* 172400 relations have 3 large primes
* 16 relations have 4 large primes

## Computation of class number

Structured Gauss elimination reduces the matrix to 33874 columns and 34189 rows.

Linear algebra is done with shader `spmv_blockcoo3` with batches of 16 moduli
using 55-bit primes, and block Wiedemann parameter `BLOCKM=3`.

* Each batch of 16 moduli (size 55 bits) took 65s GPU time (on average)
* Processing speed was equivalent to 10700 matrix-vector multiplications per second
* The first determinant has size 53677 and was computed using 992 moduli in 4477s

The smallest factors of the determinant are 2, 37, 443, 557, 1601, 2683. Only 1601
is a factor of the class number.

Using P-1 algorithm, larger factors of the determinant are found: 7443481, 67795253.
Prime factor 67795253 is a divisor of the class number.

Using P+1 algorithm, we find a factor 451068294059: it is also a factor of the class number.

The following binary forms have prime order in the class group
```
order 1601
Qfb(615718465470974762340536316792281240702753512175841502078,
   -469332799455695724878033084327039224868354769887506449063,
   1314913113598302604688642740639306548762964276156885809914)

order 67795253
Qfb(350900492502970989070774919316890905506110510480063489226,
    232243655765542440239868089700674799743259086070357175211,
   2188746555479717725685931317484552383893127174806941935555)

order 451068294059
Qfb(896926469681986261435806585038105414411771818223393852809,
   -621900414173347907775380426625023046232831526261545278955,
    949061071818114810014611169968515846060659484179076502734)
```

Note that 37 is also a factor of the class number, but ideals `[2]` and `[3]`
are in the index 37 subgroup.

It is expected that other factors would take more than 1 hour to find using ECM.

The second determinant has size 53679 bits and was computed using 992 moduli
in 4516 seconds.

The class number is 6289291536938678379602129540707019057698830932083293714973
or 0x1007f4478ae67a20b38152e98c741e041493d32596c7a7a1d in base 16, which is
very close to $2^{192}$ as intended.

The class group is cyclic and generated by ideal `[5]`.

The total computation time is 70 minutes for sieve and 150 minutes for linear algebra
(suggesting that chosen parameters were suboptimal).

## Additional examples

More examples can be generated easily:

```
p = 0x139025fc574e848871d716f82a6f73755b29611a516a6022ea3a2a1aa7a4551918761c509811022708a62f56dc93bf17
p = 3011033532210910799418391771577945094022961970202012367960304315262012434433127847314448991678943667087125949693719
p + 1 = 2^3 * 3 * 5 * 7 * 11^2 * 13^2 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
* 47 * 53 * 59 * 61 * 67 * 71 * 73 * 79 * 83 * 89 * 97 * 101 * 103 * 107 * 109
* 113 * 127 * 131 * 137 * 139 * 149 * 151 * 157 * 163 * 167 * 173 * 179 * 181
* 191 * 193 * 197 * 199 * 211 * 223 * 227 * 229 * 233 * 563 * 42557 * 49424845693
α = 8.006
h = 6318908544930947206467899026556287517326367354567172308233
h = 0x101b47bc3e3966be69cc01675b46aec90fbed4631f4643d09
h = 26589264131 * 50124138881 * 1480760246232089 * 3201872841574403026427
```

```
p = 0x13911033838d5c0ec319844c2525a91531ccd44df5f5c29ace90fb1a4b66394aa97dc050b02e8fd1ef49bbcc1810beaf
p = 3011583597330552555205288197407253285192243563846264926040942821166288075595200410094066262969557496350348556680879
p + 1 = 2^4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29^2 * 31 * 37 * 41 * 43^2
* 47 * 53 * 59 * 61 * 67 * 71 * 73 * 79 * 83 * 89 * 97 * 101 * 103 * 107 * 109
* 113 * 127 * 131 * 137 * 139 * 149 * 151 * 157 * 163 * 167 * 173 * 179 * 181
* 191 * 193 * 197 * 199 * 211 * 223 * 227 * 229 * 233 * 67911549816404419
α = 7.961
h = 6297658806049511001713065558218414823306531679576022649821
h = 0x100d6a03687dae78519d63814aaf8ea14e11455f81f110bdd
h = 3^2 * 11 * 3923334455129 * 204439672776877 * 79309173445644416831497138963 (cyclic)
```

```
p = 0x13c8c458a1dd29ea638aaa123065b97ff60754a2aadfa77b1978128dd3b4d617baafa6b50f0dae9cf4e19a88670a7d97
p = 3045074155084951365194910502149926683755888078350567497515817026190331052578387930096846340481761272789350249823639
p + 1 = 2^3 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
* 47 * 53 * 59 * 61 * 67 * 71 * 73 * 79 * 83 * 89 * 97 * 101 * 103 * 107 * 109
* 113 * 127 * 131 * 137 * 139 * 149 * 151 * 157 * 163 * 167 * 173 * 179 * 181
* 191 * 193 * 197 * 199 * 211 * 223 * 227 * 229 * 233 * 1613 * 35390558707262947
α = 7.897
h = 6312012382524566416539225793087259273238137008773421984253
h = 0x1016c7bf217dcf397d6becddfff92245f507de1ffe985e9fd
h = 9567451 * 659738145774100794092305860054810761323798471376903
```

```
p = 0x13a4b44c5bd83e30ce85d10efb64d9691bab8b51e4d14a0d4022c6eb31cbd1e8b7f627df4cfdcfd0f66170f43d774aef
p = 3023392297363646824505865242066373264311199312289631977173383319481785637071383340299943960045157623266100536363759
p + 1 = 2^4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
* 53 * 59 * 61 * 67 * 71 * 73^2 * 79 * 83 * 89 * 97 * 101 * 103 * 107 * 109
* 113 * 127 * 131 * 137 * 139 * 149 * 151 * 157 * 163 * 167 * 173 * 179 * 181
* 191 * 193 * 197 * 199 * 211 * 223 * 227 * 229 * 233 * 2593 * 2927 * 153448121087
α = 7.965
h = 6314053657099332737155067726726688464053409845087362946209
h = 0x10181cbcad5b9484a6eaa5da21d66ab2b3de0b43ec6e914a1
h = 3 * 33149 * 10548271 * 393375457 * 15301289867130494521157068549908047401
```

The most effective sieving parameters are B1=700k B2=21M, allowing each computation
to run in about 2.5 hours on a Ryzen 7840HS APU.
