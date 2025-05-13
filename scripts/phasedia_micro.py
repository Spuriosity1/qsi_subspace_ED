from ringflip_hamiltonian import RingflipHamiltonian, calc_polarisation, get_group_characters
import pyrochlore
import sys
import numpy as np
import argparse
from db import connect_npsql, init_db, rotation_matrices
import os
from phasedia_impl import calc_ring_exp_vals


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


def assert_unif_sector(rfh):
    sector = calc_polarisation(rfh.lattice, rfh.basis[0])
    if not all(calc_polarisation(rfh.lattice, b) == sector for b in rfh.basis):
        print("WARN: basis is not polarization-split")
        sector = None
    print("SECTOR: " + str(sector))
    return sector


if __name__ == "__main__":

    ap = get_parser()
    a = ap.parse_args()

    lat = pyrochlore.import_json(a.lattice_file)
    g_111 = g_111_dict[a.rotation]
    g_110 = g_110_dict[a.rotation]

    # ugly hack to keep the rotated versions distinct
    latvecs = rotation_matrices[a.rotation] @ np.array(lat.lattice_vectors)

    rfh = RingflipHamiltonian(lat, pyrochlore.get_rings)
    print("Setting up basis...")
    if a.basis_file is None:
        bfile = rfh.basisfile_loc
    else:
        bfile = a.basis_file
    rfh.load_basis(bfile)

    DB_FILE = a.db_file

    initialise = True
    if os.path.isfile(DB_FILE):
        print("WARN: db already exists!")
        if a.edit_existing:
            initialise = False
        else:
            sys.exit(1)

    con = connect_npsql(DB_FILE, timeout=60)
    if initialise:
        init_db(con)

    k_sectors = get_group_characters(rfh.lattice)

    sector = assert_unif_sector(rfh)


    if not a.no_calc_111:
        print("111 field:")
        for sign in [-1, 1]:
            for K in k_sectors:
                e, O = calc_ring_exp_vals(rfh, g=sign*g_111(a.x),
                                          krylov_dim=a.krylov_dim,k_sector=K)
                cursor = con.cursor()
                cursor.execute("""INSERT INTO field_111 (g0_g123, g123_sign, latvecs,
                                                         sector,
                                                         kx, ky, kz, edata,
                                                         O0, O1, O2, O3
                                                         )
                               VALUES (?,?,?,?,
                                       ?,?,?,?,
                                       ?,?,?,?
                                       );""",
                               (a.x, sign, latvecs, str(sector), K[0], K[1], K[2], np.real(e), *O.values() ))
            cursor.close()
            con.commit()

    if not a.no_calc_110:
        print("110 field:")
        for sign in [-1, 1]:
            for K in k_sectors:
                e, O = calc_ring_exp_vals(rfh, g=sign*g_110(a.x),
                                          krylov_dim=a.krylov_dim, k_sector=K)

                cursor = con.cursor()

                cursor.execute("""INSERT INTO field_110 (g01_g23, g23_sign, latvecs, sector,
                                                             kx, ky, kz, edata,
                                                             O0, O1, O2, O3
                                                             )
                                   VALUES (?,?,?,?,
                                           ?,?,?,?,
                                           ?,?,?,?
                                           );""",
                               (a.x, sign, latvecs, str(sector), K[0], K[1], K[2], np.real(e), *O.values() )
                               )

                cursor.close()
                con.commit()

                # save the data


