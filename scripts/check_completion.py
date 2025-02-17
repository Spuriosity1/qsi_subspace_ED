
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


a = ap.parse_args()


lat = pyrochlore.import_json(a.latfile)
latvecs = rotation_matrices[a.rotation] @ np.array(lat.lattice_vectors)


latfile_name = os.path.basename(a.latfile)
sector_dir = os.path.join("basis_partitions/",
                          latfile_name.rsplit( ".", 1 )[ 0 ])

sectors = os.listdir(sector_dir)

con = connect_npsql(a.result_database)

for sec_str in sectors:
    sector = tuple(int(a) for a in sec_str[1:-1].split('.'))
    c = con.execute("""
        SELECT count(*) FROM field_111 WHERE latvecs=? AND sector = ?
                """, (latvecs, str(sector)))
    n_records, = c.fetchone()
    c.close()
    print(f"Sector {sector}: {n_records} records")









