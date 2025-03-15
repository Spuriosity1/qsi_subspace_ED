from lattice import Lattice
import pyrochlore
import sys
import os
import argparse


parser = argparse.ArgumentParser(
    description="Generates a lattice with the specified lattice vectors")
parser.add_argument("--a1", nargs=3, type=int,
                    help="First axis of supercell (integers only)")
parser.add_argument("--a2", nargs=3, type=int,
                    help="First axis of supercell (integers only)")
parser.add_argument("--a3", nargs=3, type=int,
                    help="First axis of supercell (integers only)")

parser.add_argument("out_dir", type=str,
                    help="path to output directory (name is automatic)")

parser.add_argument("--delete_sites", type=int, nargs="+",
                    help="indices of sites to yeeto deleto")

parser.add_argument("--visualise", "-i", default=False,
                    action="store_true",
                    help="interactive mode, visualises the constructed lattice before saving")

args = parser.parse_args()

cellspec = [args.a1, args.a2, args.a3]

name = "_".join(["%d,%d,%d" % tuple(a) for a in cellspec])

lat = Lattice(pyrochlore.primitive, cellspec)

if args.delete_sites is not None:
    name += "_d"
    popped = []
    for j, s in enumerate(args.delete_sites):
        popped.append(str(s))
        assert s < lat.num_atoms, f"Invalid lattice index: {s}"
        lat.delete_atom_at_idx(s - j)

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
    visual.plot_atoms(ax, lat)
    visual.plot_bonds(ax, lat)
    fig.show()

    save_it = input(f"Save this lattice to {o_path}? [y/n] "
                    ).lower().startswith('y')

if save_it:
    pyrochlore.export_json(lat, o_path)
    print(f"Saved to {o_path}")

