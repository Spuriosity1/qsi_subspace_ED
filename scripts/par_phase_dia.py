import os
# stop numpy from getting any funny ideas
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from phasedia_impl import calc_spectrum, calc_ring_exp_vals
from tqdm import tqdm
import pyrochlore
import numpy as np
import argparse
from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
from db import connect_npsql

from multiprocessing import Pool

num_cpus=int(os.environ["SLURM_CPUS_ON_NODE"])
print(f"phasedia thinks it has {num_cpus} cpus")

ap = argparse.ArgumentParser()
ap.add_argument("lattice_file", type=str)
ap.add_argument("--db_path", type=str, default="results.db")
ap.add_argument("--min_x", "-x", type=float, default=-2)
ap.add_argument("--max_x", "-X", type=float, default=2)
ap.add_argument("--x_step", "-d", type=float, default=0.01)
ap.add_argument("--n_procs", type=int, default=num_cpus)
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


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


R3 = np.sqrt(3)
R2 = np.sqrt(2)


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




def simulate_111(sector, con):
    print("111 field:")
    for x in x_list:
        for sign in [-1, 1]:
            ring_e, ring_O = calc_ring_exp_vals(rfh, g=sign*np.array([x, 1, 1, 1]),
                                      sector=sector)


            cursor = con.cursor()
            cursor.execute("""INSERT INTO field_111 (g0_g123, g123_sign, latvecs, sector,
                                                     edata,
                                                     expO0, expO1, expO2, expO3)
                           VALUES (?,?,?,?,?,?,?,?,?);""", (x, sign, latvecs, str(sector), ring_e, *ring_O.values()))
            cursor.close()
            con.commit()

def simulate_110(sector, con):
    print("110 field:")
    for x in x_list:
        for sign in [-1, 1]:
            ring_e, ring_O = calc_ring_exp_vals(rfh, g=sign*np.array([x, x, 1, 1]),
                                      sector=sector)

            cursor = con.cursor()
            cursor.execute("""INSERT INTO field_110 (g01_g23, g23_sign, latvecs, sector,
                                                     edata,
                                                     expO0, expO1, expO2, expO3)
                           VALUES (?,?,?,?,?,?,?,?,?);""", (x, sign, latvecs, str(sector), ring_e, *ring_O.values()))

            cursor.close()
            con.commit()


def process_sector(sector):

    print("SECTOR: " + str(sector))
    con = connect_npsql(a.db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")

    simulate_111(sector, con)
    simulate_110(sector, con)
    con.close()

    return f"completed sector {sector}"




if __name__ == '__main__':
    with Pool(a.n_procs) as p:
        print(p.map(process_sector, rfh.sectors))
