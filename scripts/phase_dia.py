from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy.linalg as LA
import numpy as np
from tqdm import tqdm
import argparse
from db import connect_npsql


def calc_spectrum(g, full_lat: RingflipHamiltonian):
    results = {}
    for s in full_lat.sectors:
        H = build_matrix(full_lat, g=g, sector=s)

        if H.shape[0] < 10000:
            e, v = np.linalg.eigh(H.todense())
            results[s] = (e, v)
        else:
            e, v = sLA.eigs(H, k=100, which='SR')
            results[s] = (e, v)
    return results


def calc_ring_exp_vals(rfh: RingflipHamiltonian, g, sector, algo='sparse',
                       krylov_dim=80):
    # calculates the rimg expectation values o nthe four sublats, including
    # degeneracies
    H = build_matrix(rfh, sector=sector, g=g)

    if H.shape[0] -1 < krylov_dim:
        algo = 'dense'

    alg_opts = {
        'sparse': lambda hh: sLA.eigs(hh, k=krylov_dim, which='SR'),
        'dense': lambda hh: LA.eigh(hh.todense())
    }

    e, v = alg_opts[algo](H)

    # account for possible degenerate ground state
    mask = (e-e[0]) < 1e-10

    degen_energy = e[mask]
    degen_psi = v[:, mask]
    # print(f"degeneracy: {degen_energy.shape[0]}")
    O_list = ring_exp_values(rfh, sector, degen_psi)

    tallies = {}
    num_entries = {}
    for ring in rfh.ringflips:
        tallies[ring.sl] = 0.
        num_entries[ring.sl] = 0

    for ring, O in zip(rfh.ringflips, O_list):
        tallies[ring.sl] += O
        num_entries[ring.sl] += 1

    for k in tallies:
        if num_entries[k] > 0:
            tallies[k] /= num_entries[k]

    return tallies, e  # , degen_energy.shape[0]



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


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


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
