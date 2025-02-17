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

parser.add_argument("--visualise", default=False, type=bool, 
                    action="store_true", 
                    help="visualise the constructed lattice before saving")

args = parser.parse_args()

cellspec = [args.a1, args.a2, args.a3]

name = "_".join(["%d,%d,%d" % tuple(a) for a in cellspec])

lat = Lattice(pyrochlore.primitive, cellspec)
print(f"Modified unit cell: {lat.primitive.lattice_vectors}")
print("Periodicity: %d %d %d" % tuple(lat.periodicity))

pyrochlore.export_json(lat, os.path.join(args.out_dir, "pyro_"+name+".json"))

