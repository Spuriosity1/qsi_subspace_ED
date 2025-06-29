from lattice import Lattice, get_transl_generators
import pyrochlore
import itertools
import tqdm
import os
from bisect import bisect_left
import numpy as np
import subprocess
import time
import scipy.sparse as sp
from bit_tools import bitperm, make_state, as_mask
from scipy.optimize import curve_fit
import h5py

uint128 = np.dtype([
  ('hi', 'i8'), 
  ('lo', 'i8'),
])

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


class BasisBenchmarker:
    def __init__(self):
        self.test_latfile_dir = "test/data"
        self._popt = None

    def run(self, N, nthread):
        lengths = list(range(1,N))
        times = [self._time_test_sim(L, nthread) for L in lengths]

        self.measured = {}
        for L, t in zip(lengths, times):
            print(f"{L:3d} -> {t:.4f} seconds")
            self.measured[(1, 1, L)] = t

        def _model(L, A, t0):
            return t0 + A*np.exp(L)

        popt, pcov = curve_fit(_model, lengths, times,
                               (0.1, 0.001),
                               bounds=((0, 0), (1, np.inf))
                               )
        self._popt = popt
        self._pcov = pcov


    def estimated_runtime(self, bravais_vectors):
        n_cells = np.abs(np.linalg.det(np.array(bravais_vectors,dtype=np.float64)))
        A, t0 = self._popt
        sigma_A = np.sqrt(self._pcov[0,0])
        sigma_t0 = np.sqrt(self._pcov[1,1])
        return (
                A*np.exp(n_cells) + t0,
                (A-sigma_A)*np.exp(n_cells) + t0-sigma_t0,
                (A+sigma_A)*np.exp(n_cells) + t0+sigma_t0
                )



    def _time_test_sim(self, L, nthread):
        latfile_loc = os.path.join(
                self.test_latfile_dir,
                f"testlat_1_1_{L}.json"
                )
        lattice = Lattice(
            bravais_vectors=[[1, 0, 0], [0, 1, 0], [0, 0, L]],
            primitive_suggestion=pyrochlore.primitive,
            populate=True)
        pyrochlore.export_json(lattice, latfile_loc)

        t1 = time.time()
        subprocess.run(
            ["build/gen_spinon_basis_parallel",
             latfile_loc,
             str(nthread)])

        t2 = time.time()

        return t2 - t1



def build_ringop(rf, basis_set):

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



def load_basis_csv(fname, print_every=100):
    basis = []
    line_no = 0
    with open(fname, 'r') as f:
        for line in f:
            if not line.startswith("0x"):
                continue
            if (line_no % print_every == 0):
                print(f"\r line {line_no}", end='')
            state = int(line, 16)
            basis.append(state)

            line_no += 1
    print()
    return basis


class RingflipHamiltonian:
    """
    Responsible for storing the ringflips and the spinon-free basis.
    """

    def __init__(self, dimensions, get_rings, include_partial=False):
        if type(dimensions) is list or isinstance(dimensions, np.ndarray):
            self.lattice = Lattice(pyrochlore.primitive, dimensions)
        elif isinstance(dimensions, Lattice):
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
            raise AttributeError("basis is None, run `calc_basis` first")

    def save_basis(self, filename = None):
        if filename is None:
            filename = "basis_"+self.lattice.shape_hash()+".csv"
        self._assert_basis()
        with open(filename, 'w') as f:
            for b in self.basis:
                f.write('0x%08x\n' % b)

    def _load_basis_h5(self, fname, print_every=100):
        self.basis = []
        with h5py.File(fname, 'r') as f:
            data = f["basis"][:]  # Read full dataset

        if data.shape[1] != 2:
            raise ValueError("Dataset shape must be (N,2) for Uint128 storage.")
        
        # Convert (N,2) uint64 array to (N,) uint128 array
        for i in range(data.shape[0]):
            v = int(data[i, 0])
            v |= (int(data[i, 1]) << 64)
            self.basis.append(v)


    def load_basis(self, fname, print_every=100, sort=False):
        """
        loads the basis from file
        @param fname: the filename to load, in CSV format
        """
        print("Loading basis...\n")
        if fname.endswith('.csv'):
            self.basis = load_basis_csv(fname, print_every)
        elif fname.endswith('.h5'):
            self._load_basis_h5(fname, print_every)
        else:
            raise ValueError("Basis file must be either h5 or csv")


        if sort:
            print("\n Sorting...")
            self.basis.sort()

    @property
    def latfile_loc(self):
        name = self.lattice.shape_hash()
        return f"lattice_files/pyro_"+name+".json"

    @property
    def basisfile_loc(self):
        name = self.lattice.shape_hash()
        return f"lattice_files/pyro_"+name+".0.basis.csv"


    def generate_basis_file(self, nthread):
        subprocess.run(
            ["build/gen_spinon_basis_parallel",
             self.latfile_loc,
             str(nthread)])


    def flip_state(self, state):
        return state ^ (2**self.lattice.num_atoms - 1)

    @property
    def Hilbert_dim(self):
        self._assert_basis()
        return len(self.basis)

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


    def build_ringops(self):

        ringxc_ops = []  # ring exchange
        ring_L_ops = []  # chiral ring-flip only

        for r in self.ringflips:
            ringL = build_ringop(r, self.basis)

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



def build_matrix(h: RingflipHamiltonian, g, sigmaz=None, k_basis=None):
    """
    Builds a matrix representation of the ringflip Hamiltonian
    with respect to the basis stored in h, with ringflip coefficients g.
    There are three ways to specify g -
    1. A single numeric value (constant g)
    2. A four-member list (the four sublattices)
    3. A list of length len(h.ringxc_ops) (general)

    @param sigmaz -> a 6-member list of coeddicients of ZZ terms.
    """
    exchange_consts = np.ones(len(h.ringflips), dtype=np.float64)
    # g can be float or list of float
    if hasattr(g, "__len__"):
        assert not any(np.iscomplex(g)), "only real exchanges are supported\
        apply_ring_ham needs to be changed otherwise"
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

    ringxc_ops, _ = h.build_ringops()
    assert len(ringxc_ops) == len(exchange_consts)

    
    hO =  sum(gi * O for gi, O in zip(exchange_consts, ringxc_ops))
    if sigmaz is not None:
        for bond in h.lattice.bonds:
            hO += sigmaz[bond.color] * h.build_operator("ZZ", sites=[bond.from_idx, bond.to_idx])
            
    return hO


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
