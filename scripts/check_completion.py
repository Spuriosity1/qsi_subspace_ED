
import pyrochlore
import sqlite3
import numpy.linalg as LA
import numpy as np
from db import connect_npsql, rotation_matrices
import sys
import os
import argparse
from tqdm import tqdm

assert '.git' in os.listdir('.')

ap = argparse.ArgumentParser()
ap.add_argument("result_database", type=str,
                help="DB containing relevant output")
ap.add_argument("latfile", type=str,
                help="JSON file of lattice")
ap.add_argument("--rotation", type=str, choices='I X Y Z '.split(), default='I',
                help="Rotates lattice relative to magnetic field")
ap.add_argument("-v", "--verbosity", type=int, default=1)
ap.add_argument("--output_file", type=str, default="incomplete_sectors.txt")

a = ap.parse_args()


lat = pyrochlore.import_json(a.latfile)
latvecs = rotation_matrices[a.rotation] @ np.array(lat.lattice_vectors)


latfile_name = os.path.basename(a.latfile)
sector_dir = os.path.join("basis_partitions/",
                          latfile_name.rsplit( ".", 1 )[ 0 ])

sectors = os.listdir(sector_dir)

con = connect_npsql(a.result_database)
counts = []

for sec_str in tqdm(sectors,desc="iterating sectors"):
    sector = tuple(int(a) for a in sec_str[1:-1].split('.'))
    c = con.execute("""
        SELECT count(*) FROM field_111 WHERE latvecs=? AND sector = ?
                """, (latvecs, str(sector)))
    n_records, = c.fetchone()
    counts.append(n_records)

    c.close()
    if a.verbosity >= 2:
        print(f"Sector {sector}: {n_records} records")

expected_n_records = max(counts)

print(counts)
if min(counts) == expected_n_records:
    sys.exit(0)

damaged_sectors, = np.nonzero(np.array(counts) != expected_n_records)

print(f"Expect {expected_n_records} records")

if a.verbosity >= 1:
    print("\n\nINCOMPLETE SECTORS:")

with open(a.output_file, 'w') as f:
    for idx in damaged_sectors:
        if counts[idx] == 0:
            print( "\033[91m",end="")
        print(f"{idx:6d} Sector {sectors[idx]}: {counts[idx]} records")

        if counts[idx] == 0:
            print( "\033[0m",end="")
        f.write(f"{sectors[idx]}\n")


