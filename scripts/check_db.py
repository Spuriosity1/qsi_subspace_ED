from tqdm import tqdm
import pyrochlore
import numpy as np
from db import rotation_matrices
import os
import argparse
import phase_dia


assert '.git' in os.listdir('.')

ap = argparse.ArgumentParser()
ap.add_argument("result_database", type=str,
                help="DB containing relevant output")
ap.add_argument("planfile", type=str,
                help="Input to SLURM")
ap.add_argument("rerun_file", type=str, help="incomplete runs")
ap.add_argument("-v", "--verbosity", type=int, default=2)

a = ap.parse_args()
f = open(a.planfile, 'r')
out_file = open(a.rerun_file, 'w')

for line in f:
    sim_args = phase_dia.ap.parse_args(line.split(' '))

    lat = pyrochlore.import_json(sim_args.lattice_file)
    latvecs = rotation_matrices[sim_args.rotation] @ np.array(lat.lattice_vectors)
    sector = tuple(os.path.basename(sim_args.basis_file)[1:-1].split('.'))
    x_list = phase_dia.calc_x_list(sim_args)
    incomplete=False
    for x in tqdm(x_list):
        for sign in [-1, 1]:
            if not phase_dia.has_110_entry(x, sign, latvecs, sector):
                incomplete=True

            if not phase_dia.has_111_entry(x, sign, latvecs, sector):
                incomplete=True

    out_file.write(line)

    




f.close()
out_file.close()
