# How to inddex a general unit cell

The problem statement: Given primitive cell vectors [a1 a2 a2], and a supercell spec Z ( a 3x3 integer valued matrix), how does one index the distinct sublattices in a sane way?

Formal definitions:

Let $A = a Z$ be the supercell's periodic vectors

A "site" is a supercell-periodic lattice site ${x + A n | n \in \mathbb{Z}^3}$

A "sublattice" is a lattice of the form ${x + a n | n \in \mathbb{Z}^3}$. 

As a set, we have site $\subseteq$ sublattice.

Sublattics have the property that a *disjoint* union of finitely many sublats covers all the sites.


The indexing scheme: A site can be expressed as
$$
R = \{ a[Zn + m + r] | n \in \mathb{Z}^3 \}
$$
where r \in [0,1)^3

Choosing an arb. representative, find
a^-1 R = Zn + m + r
can calculate (a^-1 R) % 1 -> r
remainder 
x = int( (a^-1R) - r )
then do 
m = S^-1 [S@x % D]


