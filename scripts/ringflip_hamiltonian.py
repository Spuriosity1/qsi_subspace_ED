from lattice import Lattice, get_transl_generators
# import pyrochlore
from bisect import bisect_left
import numpy as np
import numpy
import scipy.sparse as sp
from bit_tools import bitperm, make_state, as_mask
import h5py
import itertools
from sympy.combinatorics import Permutation


def all_unique(ll: list):
    return len(set(ll)) == len(ll)


def calc_polarisation(lat: Lattice, state):
    polarisations = [0, 0, 0, 0]
    for i, a in enumerate(lat.atoms):
        if (state & (1 << i)) != 0:
            polarisations[int(a.sl_name)] += 1

    return tuple(polarisations)


def search_sorted(basis_set, state):
    '''
    Returns the index of state 'state'
    (interpreted as a binary sting of Sz states)
    Throws if state is not in the basis
    '''
    J = bisect_left(basis_set, state)
    if J>=len(basis_set) or basis_set[J] != state:
        return None
    return J


class RingflipHamiltonian:
    """
    Responsible for storing the ringflips and the spinon-free basis.
    """

    def __init__(self, dimensions, get_rings, include_partial=False):
        # if type(dimensions) is list or isinstance(dimensions, np.ndarray):
        #     self.lattice = Lattice(pyrochlore.primitive, dimensions)
        # el
        if isinstance(dimensions, Lattice):
            self.lattice = dimensions  # !!!! Note shallow copy!
        else:
            raise TypeError("Must specify either dimensions as a 3x3 int matrix or an explicit lattice")

        if any(x == 0 for x in self.lattice.periodicity):
            raise ValueError("Specified cell has zero volume")
        self.ringflips = get_rings(self.lattice, include_partial=include_partial)

        self.basis = None

        self._build_ringmasks()

    def _assert_basis(self):
        if self.basis is None:
            raise AttributeError("basis is None, load a basis first")

    def _load_basis_csv(self, fname, print_every=100):
        self.basis = []
        line_no = 0
        n_spins = self.lattice.num_atoms
        with open(fname, 'r') as f:
            for line in f:
                if not line.startswith("0x"):
                    continue
                if (line_no % print_every == 5):
                    print(f" line {line_no}", end='\r')
                state = int(line, 16)
                self.basis.append(state)
                line_no += 1
        print()


    def _load_basis_h5(self, fname, print_every=100):
        self.basis = []
        with h5py.File(fname, 'r') as f:
            data = f["basis"][:]  # Read full dataset

        if data.shape[1] != 2:
            raise ValueError("Dataset shape must be (N,2) for Uint128 storage.")


        for line_no in range(data.shape[0]):
            if (line_no % print_every == 0):
                print(f" line {line_no}", end='\r')

            low  = int(data[line_no, 0])  # LSB
            high = int(data[line_no, 1])  # MSB

            val = (high << 64) | low  # full 128-bit value

            # Create bitarray from binary string (as in CSV)
            # state = bitarray(val.to_bytes(16, 'big'))
            self.basis.append(val)

        print()


    def load_basis(self, fname, print_every=100, sort=False):
        """
        loads the basis from file
        @param fname: the filename to load, in CSV format
        """
        print("Loading basis...\n")
        if fname.endswith('.csv'):
            self._load_basis_csv(fname, print_every)
        elif fname.endswith('.h5'):
            self._load_basis_h5( fname, print_every)
        else:
            raise ValueError("Basis file must be either h5 or csv")


        if sort:
            print("\n Sorting...")
            self.basis.sort()


    @property
    def basis_dim(self):
        return len(self.basis)


    def flip_state(self, state):
        return state ^ UInt128(2**self.lattice.num_atoms - 1)

    @property
    def Hilbert_dim(self):
        self._assert_basis()
        return sum(len(bb) for bb in self.basis.values())

    def rings_all_separate(self) -> bool:
        # returns true if all rings have unique members
        for r in self.ringflips:
            if not all_unique(r.members):
                return False
        return True

# "private" utility funcitons, do not use these directly

    def _build_ringmasks(self):
        # builds a set of bitmasks that can be used
        # to efficiently evaluate whether the rings are flippable or not
        # specifically the test we need to do is
        # state & mask == make_state(r.members, 0b101010)
        # state & mask == make_state(r.members, 0b010101)
        for r in self.ringflips:
            if not all_unique(r.members):
                # skip all nonunique rings, they are unphysical
                r.mask = (0xffffffffffffffff, 0, 0)
                continue

            mask = as_mask(r.members)

            loc_state_l = 0
            for ij, j in enumerate(r.signs):
                loc_state_l |= (UInt128(j+1) >> 1) << ij

            state_l = make_state(r.members, loc_state_l)
            state_r = mask ^ state_l

            r.mask = (mask, state_l, state_r)

    def _build_ringop(self, rf, basis_set):

        coeffs_L = []
        to_idx_L = []
        from_idx_L = []

        mask, L, R = rf.mask
        for J, psi in enumerate(basis_set):
            tmp = psi & mask
            if tmp == L:                   # ring is flippable
                to_id = search_sorted(basis_set, psi ^ mask)
                if to_id is not None:
                    coeffs_L.append(1)
                    from_idx_L.append(J)
                    to_idx_L.append(to_id)

        dim = len(basis_set)
        opL = sp.coo_array((coeffs_L, (from_idx_L, to_idx_L)),
                           shape=(dim, dim)
                           ).tocsr()
        return opL

    def build_ringops(self):

        ringxc_ops = []  # ring exchange
        ring_L_ops = []  # chiral ring-flip only

        for r in self.ringflips:
            ringL = self._build_ringop(r, self.basis)

            ring_L_ops.append(ringL)
            ringxc_ops.append(ringL + ringL.T)

        return ringxc_ops, ring_L_ops


def get_group_characters(l: Lattice):
    # Returns a list of all allowable k-tuples
    T = [Permutation(t) for t in get_transl_generators(l)]
    BZ_size = 1.0
    for t in T:
        BZ_size *= np.float64(t.order())

    kk = [
            2.0*np.pi*np.array(m, dtype=np.float64) / BZ_size
            for m in itertools.product(*(range(t.order()) for t in T))
            ]
    
    return kk


def build_k_basis(rfh:RingflipHamiltonian, k):
    '''
    @param basis
    @param k a three-tuple of integers indexing k sectors

    Returns basis vectors (written wrt the original Sz basis) for a particular rep
    '''
    num_states = len(rfh.basis)
    seen = set()
    sym_basis = []

    T = [Permutation(t) for t in get_transl_generators(rfh.lattice)]


    for psi in rfh.basis:
        if psi in seen:
            continue

        orbit = set()
        coeff_dict = {}

        for m in itertools.product(*(range(t.order()) for t in T)):
            m = np.array(m)

            perm = T[0]**m[0] * T[1]**m[1] * T[2]**m[2]
            orb_state = bitperm(list(perm), psi)
            orbit.add(orb_state)

            idx = search_sorted(rfh.basis, orb_state)
            if idx is None:
                raise ValueError("Basis does not repect the symmetry.")
            coeff_dict[idx] = np.exp(1.0j*np.dot(m, k))

        for s in orbit:
            seen.add(s)

        if coeff_dict:
            idxs, vals = zip(*coeff_dict.items())
            vals = np.array(vals)
            norm = np.linalg.norm(vals)
            if norm > 1e-12:
                vec = sp.csr_matrix((vals / norm, ([0]*len(idxs), idxs)),
                                    shape=(1, num_states))
                sym_basis.append(vec)

    return sp.vstack(sym_basis)


#def build_symmetric_basis(basis, group_perms, characters):
#    '''
#    @param basis -> a list of int understood as basis elements in the e.g. Sz basis
#    @param group_perms -> a list of np arrays undersood as bit permutations
#
#    Returns basis vectors (written wrt the original Sz basis) for a particular rep
#    '''
#    num_states = len(basis)
#    seen = set()
#    sym_basis = []
#
#    for psi in basis:
#        if psi in seen:
#            continue
#
#        orbit = set()
#        coeff_dict = {}
#
#        for c, g in zip(characters, group_perms):
#            orb_state = bitperm(g, psi)
#            orbit.add(orb_state)
#            idx = search_sorted(basis, orb_state)
#            if idx is None:
#                raise ValueError("Basis does not repect the symmetry.")
#
#        for s in orbit:
#            seen.add(s)
#
#        if coeff_dict:
#            idxs, vals = zip(*coeff_dict.items())
#            vals = np.array(vals)
#            norm = np.linalg.norm(vals)
#            if norm > 1e-12:
#                vec = sp.csr_matrix((vals / norm, ([0]*len(idxs), idxs)),
#                                    shape=(1, num_states))
#                sym_basis.append(vec)
#
#    return np.vstack(sym_basis)


def build_matrix(h: RingflipHamiltonian, g, k_basis=None):
    """
    Builds a matrix representation of the ringflip Hamiltonian
    with respect to the basis stored in h, with ringflip coefficients g.
    There are three ways to specify g -
    1. A single numeric value (constant g)
    2. A four-member list (the four sublattices)
    3. A list of length len(h.ringxc_ops) (general)

    @param k_basis -> a (dim_k, dim_h) shaped matrix of the k-adapted basis vectors
    """
    exchange_consts = np.ones(len(h.ringflips), dtype=np.float64)
    # g can be float or list of float
    if hasattr(g, "__len__"):
        if len(g) == len(h.ringflips):
            exchange_consts = g
        elif len(g) == 4:
            # N.B. Case that exchange_consts is len 4 -> same behaviour
            for j, r in enumerate(h.ringflips):
                exchange_consts[j] = g[r.sl]
        else:
            raise NotImplementedError("g is not specified correctly")
    elif callable(g):
        # assume g(r: Ring)
        exchange_consts = [g(r) for r in h.ringflips]
    else:
        # assume numeric type
        exchange_consts = g * np.ones(len(h.ringflips),
                                      dtype=np.float64)


    _, ringL_ops = h.build_ringops()
    assert len(ringL_ops) == len(exchange_consts)
    if k_basis is None:
        H1 = sum(gi * O for gi, O in zip(exchange_consts, ringL_ops))
        return H1 + H1.conj().T
    else:
        # Transform the full Hamiltonian into the k-basis: H_k = V @ H @ V†
        dim_k = k_basis.shape[0]
        # dim = k_basis.shape[1]

        # Initialize Hamiltonian in momentum basis
        H_k = sp.csr_matrix((dim_k, dim_k), dtype=np.complex128)

        for gi, O in zip(exchange_consts, ringL_ops):
            # O is dim x dim, real sparse matrix in position basis
            # k_basis is dim_k x dim (unitary matrix-like, real or complex)
            # Transform: V H V† = k_basis @ O @ k_basis.conj().T
            Hk_piece = k_basis @ (O @ k_basis.conj().T)
            H_k += gi * Hk_piece

        return H_k + H_k.conj().T



def ring_exp_values(H: RingflipHamiltonian, psi, k_basis=None):
    """
    calculates <psi| O + Odag |psi> for all ringflips involved
    psi is one or several wavefunctions.
    k_basis is a (n_k by dimH) basis spec
    """
    tallies = []

    if k_basis is None:
        for ring_O, ring_L in zip(*H.build_ringops()):
            tallies.append(psi.conj().T @ ring_L @ psi)
    else:
        for ring_O, ring_L in zip(*H.build_ringops()):
            tallies.append(psi.conj().T @ k_basis @ ring_L @ k_basis.conj().T @ psi)

    return tallies
