from lattice import Lattice, get_transl_generators
import pyrochlore
import itertools
import tqdm
import os
from bisect import bisect_left
import numpy as np
import subprocess
import scipy.sparse as sp
from bit_tools import bitperm, make_state, as_mask


# inefficient implementation

def calc_spinonfree_basis(lat: Lattice):
    # input -> a Lattice object
    # output -> Sz, a set of bitstrings with spins corresponding to the order of lat.atoms; the basis vectors

    N_ATOMS = len(lat.atoms)
    N_UP_TETRAS = N_ATOMS // 4
    assert N_ATOMS % 4 == 0, "natoms must be a multiple of 4"
    T_up, T_down = pyrochlore.get_tetras(lat)
    # strategy: can tile the full lattice using up tetras
    # iterate exhaustively through the set of up tetras configs,
    # reject those that make spinons on down tetras
    states_2I2O = [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100]
    basis = []

    B_masks = np.array([as_mask(tB.members) for tB in T_down])

    for tetra_states in tqdm(itertools.product(*[
            states_2I2O]*N_UP_TETRAS), total=6**N_UP_TETRAS):
        state = 0
        for t, t_state in zip(T_up, tetra_states):
            state |= make_state(t.members, t_state)

        good = True
        # filter out B spinons
        for tB_mask in B_masks:
            if (state & tB_mask).bit_count() != 2:
                good = False
                break
        if good:
            basis.append(state)

    return sorted(basis)


def all_unique(l: list):
    return len(set(l)) == len(l)


def calc_polarisation(lat: Lattice, state):
    polarisations = [0, 0, 0, 0]
    for i, a in enumerate(lat.atoms):
        if (state & (1 << i)) != 0:
            polarisations[int(a.sl_name)] += 1

    return tuple(polarisations)


def search_sorted(basis_set, state):
    '''
    Returns the index of state 'state' (interpreted as a binary sting of Sz states)
    Throws if state is not in the basis
    '''
    J = bisect_left(basis_set, state)
    if basis_set[J] != state:
        raise LookupError(f"search_sorted is broken-> got {state:b}, best I could do was {basis_set[J]:b}")
    return J


class RingflipHamiltonian:
    """
    Responsible for storing the ringflips and the spinon-free basis.
    """

    def __init__(self, dimensions):
        if type(dimensions) is list or isinstance(dimensions,np.ndarray):
            self.lattice = Lattice(pyrochlore.primitive, dimensions)
        elif isinstance(dimensions, Lattice):
            self.lattice = dimensions  # !!!! Note shallow copy!
        else:
            raise TypeError("Must specify either dimensions as a 3x3 int matrix or an explicit lattice")

        if any( x == 0 for x in self.lattice.periodicity):
            raise ValueError("Specified cell has zero volume")
        self.ringflips = pyrochlore.get_ringflips(self.lattice)

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

    def load_basis(self, fname):
        print("Loading basis...\n")
        self.basis = {}
        line_no = 0
        with open(fname, 'r') as f:
            for line in f:
                print(f"\r{len(self.basis)} sectors | line {line_no}",end='')
                if not line.startswith("0x"):
                    continue
                state = int(line, 16)
                sector = calc_polarisation(self.lattice, state)
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
        return "lattice_files/pyro_"+name+".json"

    @property
    def basisfile_loc(self):
        name = self.lattice.shape_hash()
        return "lattice_files/pyro_"+name+".basis.csv"

    @property
    def basis_dim(self):
        return sum(len(b) for b in self.basis.values())

    def calc_basis(self, nthread=1, recalc=False):
        pyrochlore.export_json(self.lattice, self.latfile_loc)

        if recalc:
            print("Generating basis...")
            subprocess.run(
                ["build/gen_spinon_basis_parallel",
                 self.latfile_loc,
                 str(nthread)])

        if not os.path.exists(self.basisfile_loc):
            raise Exception(
                "No basis file has been generated (expected " +
                self.basisfile_loc + "). run with recalc=True to generate")

        self.load_basis(self.basisfile_loc)

        print(f"Basis stats: {len(self.basis)} charge sectors, total dim {self.basis_dim}")


    def flip_state(self, state):
        return state ^ (2**self.lattice.num_atoms - 1)

    def split_basis_irreps(self):
        # partitions the Hilbert space into irreps of the translational symmetries
        T = get_transl_generators(self.lattice)


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
                r.mask = (0xffffffffffffffff, 0, 0)
                continue

            mask = as_mask(r.members)
            state_l = make_state(r.members, 0b101010)
            state_r = make_state(r.members, 0b010101)

            r.mask = (mask, state_l, state_r)

    def _build_ringop(self, rf, basis_set):

        coeffs_L = []
        to_idx_L = []
        from_idx_L = []

        mask, L, R = rf.mask
        for J, psi in enumerate(basis_set):
            tmp = psi & mask
            if tmp == L:                   # ring is flippable
                coeffs_L.append(1)
                from_idx_L.append(J)
                to_idx_L.append(search_sorted(basis_set, psi ^ mask))

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


def build_matrix(h: RingflipHamiltonian, g, sector):
    """
    Builds a matrix representation of the ringflip Hamiltonian
    with respect to the basis stored in h, with ringflip coefficients g.
    There are three ways to specify g -
    1. A single numeric value (constant g)
    2. A four-member list (the four sublattices)
    3. A list of length len(h.ringxc_ops) (general)
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
    else:
        # assume numeric type
        exchange_consts = g * np.ones(len(h.ringflips),
                                      dtype=np.float64)

    ringxc_ops, _ = h.build_ringops(sector)
    assert len(ringxc_ops) == len(exchange_consts)

    return sum(gi * O for gi, O in zip(exchange_consts, ringxc_ops))


def ring_exp_values(H: RingflipHamiltonian, sector, psi):
    """
    calculates <psi| O + Odag |psi> for all ringflips involved
    psi is one or several wavefunctions.
    """
    tallies = []

    ringxc_ops, _ = H.build_ringops(sector)

    for ring_O in ringxc_ops:
        tallies.append(psi.conj().T @ ring_O @ psi)

    return tallies



