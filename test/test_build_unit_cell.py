import lattice
import visual
import matplotlib.pyplot as plt
import numpy as np
from sympy import Rational, Matrix

primitive_fcc = lattice.PrimitiveCell([[0, 4, 4],
                                 [4, 0, 4],
                                 [4, 4, 0]])

primitive = lattice.reshape_primitive_cell(primitive_fcc,
                                         #  Matrix([[1,0,0],[0,1,0],[0,0,1]]))
                                           Matrix([[1,1,0],[0,1,1],[0,0,1]]))


disp = [
    Matrix(v) for v in [
        [0, 0, 0],
        [0, 2, 2],
        [2, 0, 2],
        [2, 2, 0]
    ]]

a = [primitive.add_sublattice(str(j), disp[j]+Matrix([8,8,-8])) for j in range(4)]

# six bond sublattices, store them like su(4) generators as lower triang mat
c = [[lattice.Coupling("j01", np.zeros((3, 3), dtype=np.complex128))
      for j in range(i)]
     for i in range(4)]

for j, color in zip([1, 2, 3], ['r', 'g', 'b']):
    primitive.add_bond(a[0], disp[j], c[j][0], fmt={'color': color})
    primitive.add_bond(a[j], disp[j],  c[j][0], fmt={'color': color})

for (i, j), color in zip([(1, 2), (2, 3), (3, 1)], ['r', 'g', 'b']):
    delta = disp[j] - disp[i]
    primitive.add_bond(a[i], delta,  c[j][0], fmt={'color': color})
    primitive.add_bond(a[i], -delta,  c[j][0], fmt={'color': color})

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
visual.plot_cell(ax, primitive)

plt.show()
