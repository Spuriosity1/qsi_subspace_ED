from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy.linalg as LA
import numpy as np
from tqdm import tqdm
import argparse
from db import connect_npsql, init_db, rotation_matrices
import os
from phasedia_impl import calc_spectrum, calc_ring_exp_vals
import time

def get_parser():
    ap = argparse.ArgumentParser(prog="PHASE_DIA")
    ap.add_argument("lattice_file", type=str)
    ap.add_argument("--min_x", '-x', type=float, default=-2)
    ap.add_argument("--max_x", '-X', type=float, default=2)
    ap.add_argument("--x_step", '-d', type=float, default=0.01)
    ap.add_argument("--basis_file", type=str, default=None)
    ap.add_argument("--rotation", type=str, choices='I X Y Z '.split(), default='I',
                    help="Rotates lattice relative to magnetic field")
    ap.add_argument("--spacing", choices=["linear", "log", "tanh"], default="linear")
    ap.add_argument("--db_repo", type=str, default="./",
                    help="Directory to store results in")
    ap.add_argument("--index", type=int, default=time.time_ns())
    ap.add_argument("--krylov_dim", type=int, default=200)

    return ap


g_111_dict = {
        'I': lambda x : np.array([x, 1, 1, 1]),
        'X': lambda x : np.array([1, x, 1, 1]),
        'Y': lambda x : np.array([1, 1, x, 1]),
        'Z': lambda x : np.array([1, 1, 1, x])
        }


g_110_dict = {
        'I': lambda x : np.array([x, x, 1, 1]),
        'X': lambda x : np.array([x, x, 1, 1]),
        'Y': lambda x : np.array([1, 1, x, x]),
        'Z': lambda x : np.array([1, 1, x, x])
        }



def calc_x_list(args):
    n_points = int(np.ceil((args.max_x - args.min_x) / args.x_step))

    SPACINGS = {
        "linear": np.arange(args.min_x, args.max_x, args.x_step),
        "log": np.logspace(args.min_x, args.max_x, n_points),
        "tanh": np.tan(
             np.linspace(np.atan(args.min_x), np.atan(args.max_x), n_points)
            )
        }


    return SPACINGS[args.spacing]


tol=1e-4


def has_111_entry(con, x, sign, latvecs, sector):
    cursor = con.cursor()
    cursor.execute("""SELECT g0_g123 FROM field_111 WHERE
                    g0_g123 BETWEEN ? AND ?
                   AND g123_sign=? AND latvecs=? AND sector=?
    """, (x-tol, x+tol, sign, latvecs, str(sector))
    )
    res = cursor.fetchall() != 0
    cursor.close()
    return res


def has_110_entry(con, x, sign, latvecs, sector):

    cursor = con.cursor()
    cursor.execute("""SELECT g01_g23 FROM field_110 WHERE
                    g01_g23 BETWEEN ? AND ?
                   AND g23_sign=? AND latvecs=? AND sector=?
    """, (x-tol, x+tol, sign, latvecs, str(sector))
    )
    res = cursor.fetchall() != 0
    cursor.close()
    return res


if __name__ == "__main__":

    ap = get_parser()
    a = ap.parse_args()

    lat = pyrochlore.import_json(a.lattice_file)
    g_111 = g_111_dict[a.rotation]
    g_110 = g_110_dict[a.rotation]


    latvecs = rotation_matrices[a.rotation] @ np.array(lat.lattice_vectors)

    rfh = RingflipHamiltonian(lat)
    print("Setting up basis...")
    if a.basis_file is None:
        bfile = rfh.basisfile_loc
    else:
        bfile = a.basis_file
    rfh.load_basis(bfile)


    R3 = np.sqrt(3)
    R2 = np.sqrt(2)


    DB_FILE = os.path.join(a.db_repo, f"results_{a.index}.db")

    initialise=True
    if os.path.isfile(DB_FILE):
        print("WARN: db already exists!")
        initialise=False

    con = connect_npsql(DB_FILE, timeout=60)
    if initialise:
        init_db(con)

    x_list = calc_x_list(a)
    print(x_list)

    for sector in rfh.sectors:
        print("SECTOR: " + str(sector))

        print("111 field:")
        for x in tqdm(x_list):
            for sign in [-1, 1]:
                if has_111_entry(con, x, sign, latvecs, sector):
                    print(f"WARN: duplicate found at {x}")
                    continue

                
                r111 = calc_ring_exp_vals(rfh, g=sign*g_111(x),
                                          sector=sector, krylov_dim=a.krylov_dim)

                cursor = con.cursor()
                cursor.execute("""INSERT INTO field_111 (g0_g123, g123_sign, latvecs, sector,
                                                         edata,
                                                         expO0, expO1, expO2, expO3)
                               VALUES (?,?,?,?,?,?,?,?,?);""", (x, sign, latvecs, str(sector), r111[0], *r111[1].values()))
                cursor.close()
                con.commit()

        print("110 field:")
        for x in tqdm(x_list):
            for sign in [-1, 1]:
                if has_110_entry(con, x,sign,latvecs,sector):
                    print(f"WARN: skipping duplicate at g01_g23={x}")
                    continue

                r110 = calc_ring_exp_vals(rfh, g=sign*g_110(x),
                                          sector=sector, krylov_dim = a.krylov_dim)
                
                cursor = con.cursor()
                cursor.execute("""INSERT INTO field_110 (g01_g23, g23_sign, latvecs, sector,
                                                         edata,
                                                         expO0, expO1, expO2, expO3)
                               VALUES (?,?,?,?,?,?,?,?,?);""", (x, sign, latvecs, str(sector), r110[0], *r110[1].values()))

                cursor.close()
                con.commit()

                # save the data
