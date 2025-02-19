
import pyrochlore
import sqlite3
import numpy.linalg as LA
import numpy as np
from db import connect_npsql, rotation_matrices
import sys
import os
import argparse

assert '.git' in os.listdir('.')

ap = argparse.ArgumentParser()
ap.add_argument("result_database", type=str,
                help="DB containing relevant output")
ap.add_argument("latfile", type=str,
                help="JSON file of lattice")
ap.add_argument("--rotation", type=str, choices='I X Y Z '.split(), default='I',
                help="Rotates lattice relative to magnetic field")
ap.add_argument("-v", "--verbosity", type=int, default=2)

a = ap.parse_args()


lat = pyrochlore.import_json(a.latfile)
latvecs = rotation_matrices[a.rotation] @ np.array(lat.lattice_vectors)


latfile_name = os.path.basename(a.latfile)
sector_dir = os.path.join("basis_partitions/",
                          latfile_name.rsplit( ".", 1 )[ 0 ])

sectors = os.listdir(sector_dir)

con = connect_npsql(a.result_database)
missing = []

for sec_str in sectors:
    sector = tuple(int(a) for a in sec_str[1:-1].split('.'))
    c = con.execute("""
        SELECT count(*) FROM field_111 WHERE latvecs=? AND sector = ?
                """, (latvecs, str(sector)))
    n_records, = c.fetchone()
    if n_records == 0:
        missing.append((sector, sec_str))

    c.close()
    if a.v >= 2:
        print(f"Sector {sector}: {n_records} records")

if len(missing) == 0:
    sys.exit(0)


if a.v >= 1:
    print("\n\nMISSING SECTORS:")

with open("missing_sectors.txt", 'w') as f:
    for sec, sec_str in missing:
        print(sec)
        f.write(f"{sec_str}\n")

