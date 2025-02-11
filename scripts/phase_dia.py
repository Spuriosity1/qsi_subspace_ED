from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy.linalg as LA
import numpy as np
from tqdm import tqdm
import argparse
from db import connect_npsql
from phasedia_impl import calc_spectrum, calc_ring_exp_vals


ap = argparse.ArgumentParser()
ap.add_argument("lattice_file", type=str)
ap.add_argument("--min_x", type=float, default=-2)
ap.add_argument("--max_x", type=float, default=2)
ap.add_argument("--x_step", type=float, default=0.01)
ap.add_argument("--kappa", type=float, default=0.2,
                help="Dimensionless spacing parameter for tanh spacing")
ap.add_argument("--basis_file", type=str, default=None)
ap.add_argument("--spacing", choices=["linear", "log", "tanh"], default="linear")
a = ap.parse_args()

lat = pyrochlore.import_json(a.lattice_file)


latvecs = np.array(lat.lattice_vectors)

rfh = RingflipHamiltonian(lat)
print("Setting up basis...")
if a.basis_file is None:
    bfile = rfh.basisfile_loc
else:
    bfile = a.basis_file
rfh.load_basis(bfile)


R3 = np.sqrt(3)
R2 = np.sqrt(2)


con = connect_npsql("results.db")

con.execute("PRAGMA journal_mode=WAL")


n_points = int(np.ceil((a.max_x - a.min_x) / a.x_step))

SPACINGS = {
        "linear": np.arange(a.min_x, a.max_x, a.x_step),
        "log": np.logspace(a.min_x, a.max_x, n_points),
        "tanh": np.tan(
             np.linspace(np.atan(a.min_x),np.atan(a.max_x), n_points)
            )
        }


x_list = SPACINGS[a.spacing]
print(x_list)

for sector in rfh.sectors:
    print("SECTOR: " + str(sector))

    print("111 field:")
    for x in tqdm(x_list):
        for sign in [-1, 1]:
            r111 = calc_ring_exp_vals(rfh, g=sign*np.array([x, 1, 1, 1]),
                                      sector=sector)

            cursor = con.cursor()
            cursor.execute("""INSERT INTO field_111 (g0_g123, g123_sign, latvecs, sector,
                                                     edata,
                                                     expO0, expO1, expO2, expO3)
                           VALUES (?,?,?,?,?,?,?,?,?);""", (x, sign, latvecs, str(sector), r111[1], *r111[0].values()))
            cursor.close()
            con.commit()

    print("110 field:")
    for x in tqdm(x_list):
        for sign in [-1, 1]:
            r110 = calc_ring_exp_vals(rfh, g=sign*np.array([x, x, 1, 1]),
                                      sector=sector)

            cursor = con.cursor()
            cursor.execute("""INSERT INTO field_110 (g01_g23, g23_sign, latvecs, sector,
                                                     edata,
                                                     expO0, expO1, expO2, expO3)
                           VALUES (?,?,?,?,?,?,?,?,?);""", (x, sign, latvecs, str(sector), r110[1], *r110[0].values()))

            cursor.close()
            con.commit()

            # save the data
