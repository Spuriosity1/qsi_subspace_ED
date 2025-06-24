
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
if len(sectors) ==0:
    raise Exception(f"No basis partitions found in {sector_dir}")
sec_strings = {}
for sec_str in tqdm(sectors,desc="iterating sectors"):
    sector = tuple(int(a) for a in sec_str[1:-1].split('.'))
    sec_strings[str(sector)] = sec_str



con = connect_npsql(a.result_database)


c = con.execute("""
                SELECT sector, count(*) FROM field_111 WHERE latvecs=? GROUP BY sector
                """, (latvecs,))
counts_111 = {}

for sector, n_rec in c:
    counts_111[sector] = n_rec


c = con.execute("""
                SELECT sector, count(*) FROM field_110 WHERE latvecs=? GROUP BY sector
                """, (latvecs,))
counts_110 = {}

for sector, n_rec in c:
    counts_110[sector] = n_rec

c.close()


expected_n_records = max(counts_111.values())

if min(counts_111.values()) == expected_n_records and min(counts_110.values()) == expected_n_records:
    sys.exit(0)

damaged_sectors = []
for sec in counts_111:
    if counts_111[sec] != expected_n_records or counts_110[sec] != expected_n_records:
        damaged_sectors.append(sec)

print(f"Expect {expected_n_records} records")



if a.verbosity >= 1:
    print("\n\nINCOMPLETE SECTORS:")

with open(a.output_file, 'w') as f:
    for idx in damaged_sectors:
        if idx not in counts_111:
            counts_111[idx] = 0
        if idx not in counts_110:
            counts_110[idx] = 0

        if counts_111[idx] == 0 or counts_110[idx] == 0:
            print( "\033[91m",end="")
        print(f"Sector {idx}: {counts_111[idx]} 111 records, {counts_110[idx]} 110 records")

        print( "\033[0m",end="")
        f.write(f"{sec_strings[idx]}\n")


