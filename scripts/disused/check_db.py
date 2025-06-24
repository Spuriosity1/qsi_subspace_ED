from tqdm import tqdm
import pyrochlore
import numpy as np
from db import rotation_matrices, connect_npsql
import os
import sys
import phase_dia
import shlex

# assert '.git' in os.listdir('.')


#ap = argparse.ArgumentParser(prog="CHECK_DB")
#ap.add_argument("result_database", type=str,
#                help="DB containing relevant output")
#ap.add_argument("planfile", type=str,
#                help="Input to SLURM")
#ap.add_argument("rerun_file", type=str, help="incomplete runs")
#
#a = ap.parse_args()

rerun_file=sys.argv[3]
planfile=sys.argv[2]
f = open(planfile, 'r')
out_file = open(rerun_file, 'w')

con = connect_npsql(sys.argv[1], timeout=60)

for line in f:
    line_args = shlex.split(line)[1:]
    print(line_args)
    sim_parser = phase_dia.get_parser()
    sim_args = sim_parser.parse_args(args=line_args)

    lat = pyrochlore.import_json(sim_args.lattice_file)
    latvecs = rotation_matrices[sim_args.rotation] @ np.array(lat.lattice_vectors)
    sector = tuple(os.path.basename(sim_args.basis_file)[1:-1].split('.'))
    x_list = phase_dia.calc_x_list(sim_args)
    incomplete=False
    for x in tqdm(x_list):
        for sign in [-1, 1]:
            if not phase_dia.has_110_entry(con, x, sign, latvecs, sector):
                incomplete=True

            if not phase_dia.has_111_entry(con, x, sign, latvecs, sector):
                incomplete=True

    out_file.write(line)

    




f.close()
out_file.close()
