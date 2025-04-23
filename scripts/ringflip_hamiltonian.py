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
            raise AttributeError("basis is None, run `calc_basis` first")

    def save_basis(self, filename: str = None):
        if filename is None:
            filename = "basis_"+self.lattice.shape_hash()+".csv"
        self._assert_basis()
        with open(filename, 'w') as f:
            for sector in self.basis:
                for b in self.basis[sector]:
                    f.write('0x%08x\n' % b)

    def load_basis(self, fname, sectorfunc=None):
        """
        loads the basis from file
        @param fname: the filename to load, in CSV format
        @param sectorfunc: a funciton taking two arguments (lat and state)
            returning an immutable type. The basis is stored as a dict
            with the outputs of sectorfunc treated as keys.
        """
        if sectorfunc is None:
            def sectorfunc(lat, state): return 0
        print("Loading basis...\n")
        self.basis = {}
        line_no = 0
        with open(fname, 'r') as f:
            for line in f:
                print(f"\r{len(self.basis)} sectors | line {line_no}", end='')
                if not line.startswith("0x"):
                    continue
                state = int(line, 16)
                sector = sectorfunc(self.lattice, state)
                if sector in self.basis:
                    self.basis[sector].append(state)
                else:
                    self.basis[sector] = [state]

                line_no += 1
        print("\n Sorting...")
        for k in self.basis:
            self.basis[k].sort()

    @property
    def latfile_loc(self):
        name = self.lattice.shape_hash()
        N = self.lattice.num_atoms
        return f"lattice_files/pyro{N}_"+name+".json"

    @property
    def basisfile_loc(self):
        name = self.lattice.shape_hash()
        return f"lattice_files/pyro_"+name+".0.basis.csv"

    @property
    def basis_dim(self):
        return sum(len(b) for b in self.basis.values())

    def generate_basis_file(self, nthread):
        subprocess.run(
            ["build/gen_spinon_basis_parallel",
             self.latfile_loc,
             str(nthread)])

    def calc_basis(self, nthread=1, recalc=True, force_recalc=False,
                   sectorfunc=None):
        pyrochlore.export_json(self.lattice, self.latfile_loc)

        if recalc is True:
            if os.path.exists(self.basisfile_loc):
                print("Basis file already generated.")
            else:
                print("Generating basis...")
                self.generate_basis_file(nthread)

        if os.path.exists(self.basisfile_loc):
                print(f"Importing {self.basisfile_loc}...")
        else:
            raise Exception(
                "No basis file found (expected " +
                self.basisfile_loc + "). run with recalc=True to generate")

        self.load_basis(self.basisfile_loc, sectorfunc)

        print(f"Basis stats: {len(self.basis)} sectors, total dim {self.basis_dim}")

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

    @property
    def sectors(self):
        return self.basis.keys()

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

    def build_ringops(self, sector):

        ringxc_ops = []  # ring exchange
        ring_L_ops = []  # chiral ring-flip only

        for r in self.ringflips:
            ringL = self._build_ringop(r, self.basis[sector])

            ring_L_ops.append(ringL)
            ringxc_ops.append(ringL + ringL.T)

        return ringxc_ops, ring_L_ops

    def build_operator(self, sector, opstring: str, sites: list):
        basis_set = self.basis[sector]
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



def build_matrix(h: RingflipHamiltonian, g, sector, nk=None, k_basis=None):
    """
    Builds a matrix representation of the ringflip Hamiltonian
    with respect to the basis stored in h, with ringflip coefficients g.
    There are three ways to specify g -
    1. A single numeric value (constant g)
    2. A four-member list (the four sublattices)
    3. A list of length len(h.ringxc_ops) (general)

    @param sector -> a key to the internal 'sector' disctionary.
    @param nk -> the point in k-space, specified as three integers (in physical units, k_real = 2pi nk_j b^_j/L_j)
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

    ringxc_ops, _ = h.build_ringops(sector)
    assert len(ringxc_ops) == len(exchange_consts)

    if nk == None:
        return sum(gi * O for gi, O in zip(exchange_consts, ringxc_ops))
    else:

        raise NotImplementedError("I haven't written this")
        # Job 1: build symmetry adapted exchange operators
        assert len(g) == 4, "Cannot k-space diagonalise when translation symmetry is broken"



def ring_exp_values(H: RingflipHamiltonian, sector, psi, include_imag=False):
    """
    calculates <psi| O + Odag |psi> for all ringflips involved
    psi is one or several wavefunctions.
    """
    tallies = []
    tallies_im = []


    for ring_O, ring_L in zip(*H.build_ringops(sector)):
        tallies.append(psi.conj().T @ ring_O @ psi)
        if include_imag:
            tallies_im.append(1j*psi.conj().T @ (ring_L-ring_L.T) @ psi)

    if include_imag:
        return tallies, tallies_im
    else:
        return tallies
