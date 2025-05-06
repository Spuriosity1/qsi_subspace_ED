from ringflip_hamiltonian import RingflipHamiltonian, build_matrix
from ringflip_hamiltonian import ring_exp_values, build_k_basis
import scipy.sparse.linalg as sLA
import numpy.linalg as LA
import numpy as np


def calc_spectrum(g, full_lat: RingflipHamiltonian):
    H = build_matrix(full_lat, g=g)

    if H.shape[0] < 1000:
        e, v = np.linalg.eigh(H.todense())
        results = (e, v)
    else:
        e, v = sLA.eigs(H, k=np.min(100,H.shape[0]), which='SR')
        results = (e, v)
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






def calc_ring_exp_vals(rfh: RingflipHamiltonian, g, algo='sparse',
                       krylov_dim=200, k_sector=None):
    '''calculates the ring expectation values on the four sublats, including
    degeneracies
    @param rfh -> the ringflip Hamiltonian
    @param g -> one, four, or len(h.ringxc_ops) real g values
    @param algo -> one of  {'sparse', 'dense'} 
        (note that 'dense' is automatically selected if the Hilbert space is small)
    @param krylov_dim -> Maximum number of eigenvalues to store (dense provlems are trucnated to this)
    @param k_sector -> None, or a three-tuple of integers specifying a k sector
    '''

    if k_sector is None:
        k_basis = None
        H = build_matrix(rfh, g=g, k_basis=None)
    else:
        k_basis =build_k_basis( rfh, np.array(k_sector))
        H = build_matrix(rfh, g=g, k_basis=k_basis)

    # force dense algorithm for small problems
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
    O_list = ring_exp_values(rfh, degen_psi, k_basis)

    sum_O = {}
    num_entries = {}
    for ring in rfh.ringflips:
        sum_O[ring.sl] = 0.
        num_entries[ring.sl] = 0

    for ring, O in zip(rfh.ringflips, O_list):
        sum_O[ring.sl] += O
        num_entries[ring.sl] += 1

    for k in sum_O:
        if num_entries[k] > 0:
            sum_O[k] /= num_entries[k]

    return e, sum_O # , degen_energy.shape[0]




