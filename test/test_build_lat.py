import lattice
import visual
import matplotlib.pyplot as plt
import numpy as np
from sympy import Rational, Matrix

primitive = lattice.PrimitiveCell([[0, 4, 4],
                                 [4, 0, 4],
                                 [4, 4, 0]])



disp = [
    Matrix(v) for v in [
        [0, 0, 0],
        [0, 2, 2],
        [2, 0, 2],
        [2, 2, 0]
    ]]

sublat = [primitive.add_sublattice(
    str(j),
    disp[j]+Matrix([8,8,-8]),
    dict(color=c))
          for j, c in enumerate(['k','g','b','pink'])]

# six bond sublattices, store them like su(4) generators as lower triang mat
c = [[lattice.Coupling("j01", np.zeros((3, 3), dtype=np.complex128))
      for j in range(i)]
     for i in range(4)]

for j, color in zip([1, 2, 3], ['r', 'g', 'b']):
    primitive.add_bond(sublat[0], disp[j], c[j][0], fmt={'color': color})
    primitive.add_bond(sublat[j], disp[j],  c[j][0], fmt={'color': color})

for (i, j), color in zip([(1, 2), (2, 3), (3, 1)], ['b', 'r', 'g']):
    delta = disp[j] - disp[i]
    primitive.add_bond(sublat[i], delta,  c[j][0], fmt={'color': color})
    primitive.add_bond(sublat[i], -delta,  c[j][0], fmt={'color': color})

full_lat = lattice.Lattice(primitive, [[-1,1,1],[1,-1,1],[1,1,-1]])


fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
visual.plot_cell(ax, full_lat)

plt.show()
