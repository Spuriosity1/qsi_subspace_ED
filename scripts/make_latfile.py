from lattice import Lattice
import pyrochlore
import sys
import os


def print_error():
    print(f"USAGE: {sys.argv[0]} a1,a2,a3_b1,b2,b3_c1,c2,c3 out_dir")
    sys.exit(1)


if len(sys.argv) != 3:
    print_error()

cellspec = [[int(y) for y in x.split(",")]
            for x in sys.argv[1].split("_")]

name = "_".join(["%d,%d,%d" % tuple(a) for a in cellspec])
if name != sys.argv[1]:
    print_error()

lat = Lattice(pyrochlore.primitive, cellspec)
print(f"Modified unit cell: {lat.primitive.lattice_vectors}")
print("Periodicity: %d %d %d" % tuple(lat.periodicity))

pyrochlore.export_json(lat, os.path.join(sys.argv[2], "pyro_"+name+".json"))

