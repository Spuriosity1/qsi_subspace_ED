from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy.linalg as LA
import numpy as np


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

    if H.shape[0] - 1 < krylov_dim:
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

    return e, tallies  # , degen_energy.shape[0]




