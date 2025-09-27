from lattice import Lattice
import pyrochlore
import sys
import os
import argparse
import numpy.linalg as LA
from numpy import rint
import numpy as np

parser = argparse.ArgumentParser(
    description="Generates a lattice with the specified lattice vectors")

spec1 = parser.add_argument_group('stacking', 'Stacking of primitive cells to make the supercell')
spec1.add_argument("--a1", nargs=3, type=int,
                    help="First axis of supercell (integers only)")
spec1.add_argument("--a2", nargs=3, type=int,
                    help="Second axis of supercell (integers only)")
spec1.add_argument("--a3", nargs=3, type=int,
                    help="Third axis of supercell (integers only)")

spec2 = parser.add_argument_group('supercell', 'supercell spec')

spec2.add_argument("--A1", nargs=3, type=int,
                    help="First supercell vector (integers only)")
spec2.add_argument("--A2", nargs=3, type=int,
                    help="Second supercell vector (integers only)")
spec2.add_argument("--A3", nargs=3, type=int,
                    help="Third supercell vector (integers only)")


parser.add_argument("out_dir", type=str,
                    help="path to output directory (name is automatic)")

parser.add_argument("--delete_sites", type=int, nargs="+",
                    help="indices of sites to yeeto deleto")

parser.add_argument("--visualise", "-i", default=False,
                    action="store_true",
                    help="interactive mode, visualises the constructed lattice before saving")

args = parser.parse_args()

if args.a1:
    cellspec = [args.a1, args.a2, args.a3]
else:
# solve A = a z for z
    desired_cell = np.array([args.A1, args.A2, args.A3], dtype=int).T
    cellspec_d = LA.solve( np.array(pyrochlore.primitive.lattice_vectors,dtype=np.float64), desired_cell)
    cellspec = rint(cellspec_d)
    cellspec = [[int(i) for i in row] for row in cellspec]
    if LA.norm(cellspec_d - cellspec) > 1e-10:
        print("Bad specfification: could not find an integer presentation of unit cell")
        print(desired_cell)
        print(cellspec_d)
        raise


name = "_".join(["%d,%d,%d" % tuple(a) for a in cellspec])

lat = Lattice(pyrochlore.primitive, cellspec)

if args.delete_sites is not None:
    name += "_d"
    popped = []

    for s in reversed(sorted(args.delete_sites)):
        popped.append(str(s))
        assert s < lat.num_atoms, f"Invalid lattice index: {s}"
        lat.delete_atom_at_idx(s)

    name += ",".join(popped)

print(f"Modified unit cell: {lat.primitive.lattice_vectors}")
print("Periodicity: %d %d %d" % tuple(lat.periodicity))

save_it = True
o_path = os.path.join(args.out_dir, "pyro_"+name+".json")

if args.visualise:
    import visual
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    visual.plot_atoms(ax, lat,show_ids=True)
    visual.plot_bonds(ax, lat)
    fig.show()

    save_it = input(f"Save this lattice to {o_path}? [y/n] "
                    ).lower().startswith('y')

if save_it:
    pyrochlore.export_json(lat, o_path)
    print(f"Saved to {o_path}")

