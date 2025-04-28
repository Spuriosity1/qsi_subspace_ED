from lattice import Lattice
import pyrochlore
from bisect import bisect_left
import numpy as np
import scipy.sparse as sp
from bit_tools import bitperm, make_state, as_mask
import h5py


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

    def __init__(self, dimensions, include_partial=False):
        if type(dimensions) is list or isinstance(dimensions, np.ndarray):
            self.lattice = Lattice(pyrochlore.primitive, dimensions)
        elif isinstance(dimensions, Lattice):
            self.lattice = dimensions  # !!!! Note shallow copy!
        else:
            raise TypeError("Must specify either dimensions as a 3x3 int matrix or an explicit lattice")

        if any(x == 0 for x in self.lattice.periodicity):
            raise ValueError("Specified cell has zero volume")
        self.ringflips = pyrochlore.get_ringflips(self.lattice, include_partial=include_partial)

        self.basis = None

        self._build_ringmasks()

    def _assert_basis(self):
        if self.basis is None:
            raise AttributeError("basis is None, load a basis first")

    def _load_basis_csv(self, fname, print_every=100):
        self.basis = []
        line_no = 0
        with open(fname, 'r') as f:
            for line in f:
                if not line.startswith("0x"):
                    continue
                if (line_no % print_every == 0):
                    print(f" line {line_no}", end='')
                state = int(line, 16)
                self.basis.append(state)
                line_no += 1


    def _load_basis_h5(self, fname, print_every=100):
        self.basis = []
        with h5py.File(fname, 'r') as f:
            data = f["basis"][:]  # Read full dataset

        if data.shape[1] != 2:
            raise ValueError("Dataset shape must be (N,2) for Uint128 storage.")

        # Convert (N,2) uint64 array to (N,) uint128 array
        uint128_array = (data[:, 1].astype(np.uint128) << 64) | data[:, 0].astype(np.uint128)

        for line_no in range(data.shape[0]):
            if (line_no % print_every == 0):
                print(f" line {line_no}", end='')
            self.basis.append(uint128_array[line_no])



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
            for k in self.basis:
                self.basis[k].sort()


    @property
    def basis_dim(self):
        return sum(len(b) for b in self.basis.values())


    def flip_state(self, state):
        return state ^ (2**self.lattice.num_atoms - 1)

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
                loc_state_l |= ((j+1) >> 1) << ij

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

    def build_operator(self, opstring: str, sites: list):
        basis_set = self.basis
        assert len(set(sites)) == len(sites), "sites must not repeat"

        coeffs = []
        from_idx = []
        to_idx = []

        for J, psi in enumerate(basis_set):
            new_state = psi
            coeff = 1.0
            for j, op in zip(sites, opstring):
                if op == 'Z':
                    coeff *= 1 if (psi & (1 << j)) else -1
                elif op == 'X':
                    new_state ^= (1 << j)  # Flip spin at site j
                elif op == 'Y':
                    new_state ^= (1 << j)
                    coeff *= 1j if (psi & (1 << j)) else -1j
                elif op == '+':  # Raising operator S^+
                    if not (psi & (1 << j)):
                        new_state ^= (1 << j)
                    else:
                        coeff = 0
                        break
                elif op == '-':  # Lowering operator S^-
                    if psi & (1 << j):
                        new_state ^= (1 << j)
                    else:
                        coeff = 0
                        break

            to_id = search_sorted(basis_set, new_state)
            if abs(coeff) > 1e-11 and to_id is not None:
                coeffs.append(coeff)
                from_idx.append(J)
                to_idx.append(to_id)

        dim = len(basis_set)

        return sp.coo_array((coeffs, (from_idx, to_idx)),shape=(dim,dim)).tocsr()



def build_symmetric_basis(basis, group_perms, characters):
    '''
    @param basis -> a list of int understood as basis elements in the e.g. Sz basis
    @param group_perms -> a list of np arrays undersood as bit permutations
    '''
    num_states = len(basis)
    seen = set()
    sym_basis = []

    for psi in basis:
        if psi in seen:
            continue

        orbit = set()
        coeff_dict = {}

        for c, g in zip(characters, group_perms):
            orb_state = bitperm(g, psi)
            orbit.add(orb_state)
            idx = search_sorted(basis, orb_state)
            if idx is None:
                raise ValueError("Basis does not repect the symmetry.")

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

    return np.vstack(sym_basis)


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
        assert not any(np.iscomplex(g)), "only real exchanges are supported\
y_ring_ham needs to be changed otherwise"
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

    if k_basis is None:

        ringxc_ops, _ = h.build_ringops()
        assert len(ringxc_ops) == len(exchange_consts)
        return sum(gi * O for gi, O in zip(exchange_consts, ringxc_ops))
    else:
        # Transform the full Hamiltonian into the k-basis: H_k = V @ H @ V†
        dim_k = k_basis.shape[0]
        dim = k_basis.shape[1]

        # Initialize Hamiltonian in momentum basis
        H_k = sp.csr_matrix((dim_k, dim_k), dtype=np.complex128)

        for gi, O in zip(exchange_consts, ringxc_ops):
            # O is dim x dim, real sparse matrix in position basis
            # k_basis is dim_k x dim (unitary matrix-like, real or complex)
            # Transform: V H V† = k_basis @ O @ k_basis.conj().T
            Hk_piece = k_basis @ (O @ k_basis.conj().T)
            H_k += gi * Hk_piece

        return H_k



def ring_exp_values(H: RingflipHamiltonian, psi, include_imag=False):
    """
    calculates <psi| O + Odag |psi> for all ringflips involved
    psi is one or several wavefunctions.
    """
    tallies = []
    tallies_im = []


    for ring_O, ring_L in zip(*H.build_ringops()):
        tallies.append(psi.conj().T @ ring_O @ psi)
        if include_imag:
            tallies_im.append(1j*psi.conj().T @ (ring_L-ring_L.T) @ psi)

    if include_imag:
        return tallies, tallies_im
    else:
        return tallies
