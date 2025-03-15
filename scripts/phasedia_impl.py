from ringflip_hamiltonian import RingflipHamiltonian, build_matrix, ring_exp_values
import scipy.sparse.linalg as sLA
import pyrochlore
import numpy.linalg as LA
import numpy as np


def calc_spectrum(g, full_lat: RingflipHamiltonian):
    results = {}
    for s in full_lat.sectors:
        H = build_matrix(full_lat, g=g, sector=s)

        if H.shape[0] < 1000:
            e, v = np.linalg.eigh(H.todense())
            results[s] = (e, v)
        else:
            e, v = sLA.eigs(H, k=np.min(100,H.shape[0]), which='SR')
            results[s] = (e, v)
    return results


def eigs_retry(hh, krylov_dim):
    n = hh.shape[0]
    ncv = min(n, max(2*krylov_dim + 1, 20))
    while True:
        if n < ncv*2:
            return LA.eigh(hh.todense())
        try:
            return sLA.eigs(hh, k=krylov_dim, which='SR', ncv=ncv)

        except sLA.ArpackNoConvergence as e:
            ncv *= 2
            print("Convergence failed, restarting with enlarged Kyrlov space")
            print(f"ncv={ncv}")



def calc_ring_exp_vals(rfh: RingflipHamiltonian, g, sector, algo='sparse',
                       krylov_dim=200):
    # calculates the rimg expectation values o nthe four sublats, including
    # degeneracies
    H = build_matrix(rfh, sector=sector, g=g)

    if H.shape[0] - 2 < krylov_dim*2:
        algo = 'dense'

    alg_opts = {
            'sparse': lambda hh: eigs_retry(hh, krylov_dim),
        'dense': lambda hh: LA.eigh(hh.todense())
    }

    e, v = alg_opts[algo](H)
    e = e[:krylov_dim]
    v = v[:, :krylov_dim]

    # account for possible degenerate ground state
    mask = (e-e[0]) < 1e-10

    degen_energy = e[mask]
    degen_psi = v[:, mask]
    # print(f"degeneracy: {degen_energy.shape[0]}")
    rO_list, iO_list = ring_exp_values(rfh, sector, degen_psi, include_imag=True)

    sum_reO = {}
    sum_imO = {}
    num_entries = {}
    for ring in rfh.ringflips:
        sum_reO[ring.sl] = 0.
        sum_imO[ring.sl] = 0.
        num_entries[ring.sl] = 0

    for ring, reO, imO in zip(rfh.ringflips, rO_list, iO_list):
        sum_reO[ring.sl] += np.real(reO)
        sum_imO[ring.sl] += np.real(imO)
        num_entries[ring.sl] += 1

    for k in sum_reO:
        if num_entries[k] > 0:
            sum_reO[k] /= num_entries[k]
            sum_imO[k] /= num_entries[k]

    return e, sum_reO, sum_imO  # , degen_energy.shape[0]




