from ringflip_hamiltonian import RingflipHamiltonian, build_matrix 
from ringflip_hamiltonian import ring_exp_values, calc_polarisation
import scipy.sparse.linalg as sLA
import pyrochlore
import sys
import numpy.linalg as LA
import numpy as np
from tqdm import tqdm
import argparse
from db import connect_npsql, init_db, rotation_matrices
import os
from phasedia_impl import calc_spectrum, calc_ring_exp_vals
import time


from phase_dia import g_111_dict, g_110_dict, has_110_entry, has_111_entry


def get_parser():
    ap = argparse.ArgumentParser(prog="PHASE_DIA")
    ap.add_argument("lattice_file", type=str)
    ap.add_argument('x', type=float)
    ap.add_argument("--basis_file", type=str, default=None)
    ap.add_argument("--rotation", type=str, choices='I X Y Z '.split(), default='I',
                    help="Rotates lattice relative to magnetic field")
    ap.add_argument("--db_file", type=str, default=None,
                    help="Database to store results in")
    ap.add_argument("--edit_existing", action='store_true', default=False,
                    help='If false, refuses to modify an existing db')
    ap.add_argument("--krylov_dim", type=int, default=200)
    ap.add_argument("--no_calc_110", action='store_true', default=False)
    ap.add_argument("--no_calc_111", action='store_true', default=False)

    return ap


if __name__ == "__main__":

    ap = get_parser()
    a = ap.parse_args()

    lat = pyrochlore.import_json(a.lattice_file)
    g_111 = g_111_dict[a.rotation]
    g_110 = g_110_dict[a.rotation]

    # ugly hack to keep the rotated versions distinct
    latvecs = rotation_matrices[a.rotation] @ np.array(lat.lattice_vectors)

    rfh = RingflipHamiltonian(lat)
    print("Setting up basis...")
    if a.basis_file is None:
        bfile = rfh.basisfile_loc
    else:
        bfile = a.basis_file
    rfh.load_basis(bfile)

    DB_FILE = a.db_file

    initialise=True
    if os.path.isfile(DB_FILE):
        print("WARN: db already exists!")
        if a.edit_existing:
            initialise=False
        else:
            sys.exit(1)

    
    con = connect_npsql(DB_FILE, timeout=60)
    if initialise:
        init_db(con)


    sector = calc_polarisation(rfh.lattice, rfh.basis[0])
    for b in rfh.basis:
        s2 = calc_polarisation(rfh.lattice, b)
        if sector != s2:
            print("WARN: basis is not polarization-defined")
            sector = None
            break

    print("SECTOR: " + str(sector))
    if not a.no_calc_111:
        print("111 field:")
        for sign in [-1, 1]:
            if has_111_entry(con, a.x, sign, latvecs, sector):
                print(f"WARN: duplicate found at {a.x}")
                continue


            e, reO, imO = calc_ring_exp_vals(rfh, g=sign*g_111(a.x),
                                      sector=sector, krylov_dim=a.krylov_dim,)
            cursor = con.cursor()
            cursor.execute("""INSERT INTO field_111 (g0_g123, g123_sign, latvecs, sector,
                                                     edata,
                                                     reO0, reO1, reO2, reO3,
                                                     imO0, imO1, imO2, imO3
                                                     )
                           VALUES (?,?,?,?,
                                   ?,
                                   ?,?,?,?,
                                   ?,?,?,?);""", 
                           (a.x, sign, latvecs, str(sector), np.real(e), *reO.values(), *imO.values() ))
            cursor.close()
            con.commit()

    if not a.no_calc_110:
        print("110 field:")
        for sign in [-1, 1]:
            if has_110_entry(con, a.x,sign,latvecs,sector):
                print(f"WARN: skipping duplicate at g01_g23={a.x}")
                continue

            e, reO, imO = calc_ring_exp_vals(rfh, g=sign*g_110(a.x),
                                      sector=sector, krylov_dim = a.krylov_dim)

            cursor = con.cursor()
            cursor.execute("""INSERT INTO field_110 (g01_g23, g23_sign, latvecs, sector,
                                                     edata,
                                                     reO0, reO1, reO2, reO3,
                                                     imO0, imO1, imO2, imO3
                                                     )
                           VALUES (?,?,?,?,
                                   ?,
                                   ?,?,?,?,
                                   ?,?,?,?);""",

                           (a.x, sign, latvecs, str(sector), np.real(e), *reO.values(), *imO.values() ))


            cursor.close()
            con.commit()

            # save the data


