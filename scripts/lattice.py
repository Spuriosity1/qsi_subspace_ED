# import numpy as np
# import numpy.linalg as LA
# import scipy.sparse as sp
# import json
from sympy import Matrix, Rational, ZZ, denom, gcd, nsimplify
from sympy.matrices.normalforms import smith_normal_decomp
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import diag
import numpy as np
import itertools
import spglib

########################################
########################################
# This file contains containers that handle all the relevant geometric data.
########################################
########################################

#######################################################################
# ATOMIC STORAGE

# The most basic container, represents a single atom


class Atom:
    """
    A container storing a position only.
    """
    def __init__(self, sl_name: str, xyz):
        for x in xyz:
            if not (type(x) is int or (
                hasattr(x, 'is_rational') and x.is_rational)
            ):
                # exclude floats because of weird rounding
                raise ValueError(f"Positions must be rational, got '{type(x)}'")
        self.xyz = Matrix([Rational(x) for x in xyz])

        self.sl_name = sl_name

    def __str__(self):
        return f"Atom at ({self.xyz[0]}, {self.xyz[1]}, {self.xyz[2]}),  sublattice {self.sl_name}"

    def __repr__(self):
        return self.__str__()


class Sublattice(Atom):
    """
    A generalised atom, used only for primitive cells.
    """
    def __init__(self, sl_name: str, xyz):
        super().__init__(sl_name, xyz)
        self.bonds_from = []

    def __str__(self):
        return (f"Sublattice {self.sl_name} at {self.xyz}")

    def make_shifted_clone(self, plus: Matrix):
        '''
        Returns an Atom of the same type, offfset by 'plus'
        '''
        x = Atom(self.xyz + plus)
        return x




class Bond:
    def __init__(self, from_idx: int,
                 to_idx: int,
                 color: int,
                 bond_delta
                 ):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.color = color
        self.bond_delta = Matrix(bond_delta)

    def __repr__(self):
        return f"{self.from_idx}->{self.to_idx} color={self.color}"


def _wrap_coordinate(A: Matrix, X: Matrix):
    """
    returns 'Y' such that Y = A r, for some r in [0,1)^3
    Assumes A is a SymPy Matrix, for which inverses can be found symbolically
    """
    r = (A.inv() @ X) % 1
    return A @ r


def from_cols(*a):
    # convenience
    return nsimplify(Matrix(a).T, rational=True)


class PrimitiveCell:
    # Represents a primitive cell
    def __init__(self, a, lattice_tolerance=1e-6):
        '''
            a -> a matrix of cell vectors (understood as column vectors)
            lattice_tolerance -> currently unused
        '''
        self.sublattices = []

        a = Matrix(a)
        assert a.is_square

        self.lattice_vectors = a
        self.lattice_tolerance = lattice_tolerance

    def shape_hash(self):
        # Returns a unique identifier based on the primitive vectors
        denoms = [denom(x) for x in self.lattice_vectors]
        g_denom = gcd(denoms)
        res = ''
        for col in range(3):
            res += ",".join(["%d" % (g_denom*x) for x in self.lattice_vectors.col(col)])
            res += "b"
        res += "%d" % int(g_denom)
        return res

    @property
    def num_atoms(self):
        return len(self.sublattices)

    @property
    def atoms(self):
        return self.sublattices

    def distance(self, r1, r2):
        # check the 27 possible cell displacements
        r1 = self.wrap_coordinate(r1)
        r2 = self.wrap_coordinate(r2)
        D = []
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                for iz in [-1, 0, 1]:
                    D.append((self.lattice_vectors @
                             Matrix([ix, iy, iz])+r1-r2).norm())

        return min(D)

    def in_unitcell(self, xyz: Matrix):
        for i in range(3):
            a = self.lattice_vectors[:, i]
            proj = a.dot(xyz)
            if proj < 0 or proj > a.dot(a):
                return False
        return True

    def add_sublattice(self, sl_label: str, xyz: Matrix,
                       wrap_unitcell: bool = True):
        '''
        @param sl_label        a name for the sublattice
        @param xyz             a [3,1] vector representing the atom position
        '''
        xyz = Matrix(xyz)

        if wrap_unitcell:
            xyz = self.wrap_coordinate(xyz)
        else:
            # check that it is actually within the primitive cell
            # issue a warning if not
            if not self.in_unitcell(xyz):
                print("WARN: adding an atom outside the primtive cell")

        # check that we are not too close to anyone else
        for a in self.sublattices:
            D = self.distance(xyz, a.xyz)
            if D < self.lattice_tolerance*10:
                print(f"WARN: adding an atom very close ({D}) to existing atom {a}")

        self.sublattices.append(Sublattice(sl_label, xyz))

        if isinstance(sl_label, Sublattice):
            self.sublattices[-1].bonds_from = sl_label.bonds_from
            # indices. colors and directions are still OK

        return self.sublattices[-1]  # for further editing

    def as_sl_idx(self, xyz: Matrix):
        sl = None

        ainv = self.lattice_vectors.inv()
        for i, a in enumerate(self.sublattices):
            if all([x.is_integer for x in (ainv @ (a.xyz - xyz))]):
                sl = i
                break

        if sl is None:
            msg = f"Could not add bond: there is no site at {xyz} \n"
            msg += "Possible sites:\n"
            for x in [f"{a.sl_name} {a.xyz} \n" for a in self.sublattices]:
                msg += x + "\n"
            raise Exception(msg)

    # Bonds are always 'attached' to the 'from' index
    def add_bond(self, sl_from: Sublattice, bond_delta: Matrix,
                 color: int):
        '''
        @param sl_from     atom linking from
        @param bond_delta  a vector pointing to the next atom
        @param color      an integer, bond sublat (NOT the actual color for plotting)
        '''
        bond_delta = Matrix(bond_delta)
        to_xyz = self.wrap_coordinate(Matrix(bond_delta) + sl_from.xyz)

        sl_from.bonds_from.append(
            Bond(from_idx=self.as_sl_idx(sl_from.xyz),
                 to_idx=self.as_sl_idx(to_xyz),
                 color=color,
                 bond_delta=bond_delta)
        )


    @property
    def bonds(self):
        bonds = []
        for from_idx, a in enumerate(self.sublattices):
            for b in a.bonds_from:
                bonds.append(Bond(
                    from_idx=from_idx,
                    to_idx=b.to_idx,
                    color=b.color,
                    bond_delta=b.bond_delta,
                ))
        return bonds

    def reduced_pos(self, xyz: Matrix):
        return self.lattice_vectors.inv() @ xyz

    def wrap_coordinate(self, xyz):
        return _wrap_coordinate(self.lattice_vectors, xyz)


def reshape_primitive_cell(cell: PrimitiveCell, bravais: Matrix):
    """
    Constructs a new primitve unit cell based on the previous one, with new
    lattice vectors b1 b2 b3 constructed from A using bravais
    """
    assert all([x.is_integer for x in bravais])
    assert abs(bravais.det()) == 1
    newcell = PrimitiveCell(cell.lattice_vectors * bravais,
                            lattice_tolerance=cell.lattice_tolerance)

    new_sublats = [
        newcell.add_sublattice(
            a.sl_name, a.xyz, wrap_unitcell=True)
        for a in cell.sublattices]

    for j, a in enumerate(cell.sublattices):
        for bf in a.bonds_from:
            newcell.add_bond(
                new_sublats[j], bf.bond_delta, bf.color)

    return newcell

#######################################################
# The Lattice Objects, for storing supercells


class Lattice:
    def __init__(self, primitive_suggestion: PrimitiveCell, bravais_vectors,
                 populate=True):
        '''
         @param primitive         A primitive unit cell of the lattice. 
                                  This class may choose a different one to 
                                  better mattch the bravais vectors.
         @param bravais_vectors   The Bravais lattice vectors, in units of the primitives,
                                  of the enlarged cell. This should be a 3x3 of integers, 
                                  such that the enlarged vectors may be expressed as 
                                  (a1 a2 a3) * bravais_vectors := (b1 b2 b3)
        '''
        # set up the index scheme, define self.primitive
        self.establish_primitive_cell(
            primitive_suggestion, Matrix(bravais_vectors))
        # Populate the bonds

        self.atoms = []
        self.bonds = []  # format: {from_idx, to_idx, color}

        # XYZ hash to idx table
        self.atom_lookup = {i: None for i in range(self.primitive.num_atoms
                                    * self.periodicity[0]
                                    * self.periodicity[1]
                                    * self.periodicity[2])}
        if populate:
            self._populate_atoms()
            self._populate_bonds()

        self.shape_hash_str =  "%d_%d_%d" % tuple(self.periodicity)
        self.shape_hash_str += "x" + self.primitive.shape_hash()


    def shape_hash(self):
        return self.shape_hash_str

    def enumerate_primitives(self):
        return itertools.product(range(self.periodicity[0]),
                                 range(self.periodicity[1]),
                                 range(self.periodicity[2]))

    @property
    def lattice_vectors(self):
        # Returns a matrix of lattice vectors (understood as cols)
        return self.primitive.lattice_vectors @ diag(self.periodicity, unpack=True)

    def establish_primitive_cell(self, primitive_suggestion, bravais_vectors: Matrix):
        # Calculate the Smith decomposition of the Bravais vectors , i.e.
        # invertible S, T and diagonal D s.t.
        # S * bravais_vectors * T = D <=> bravais_vectors = S-1 D T-1
        # so can rewrite the periodicity requirement
        # (b1 b2 b3) * diag[z1, z2, z3] ~ 0
        # for all z1,z2,z3 integer
        # <=> (a1 a2 a3) * S-1 J diag(x1 x2 x3) ~ 0 for all x1 x2 x3 integer;
        # suggests a good way to index it in terms of the A1 A2 A3 := (a1 a2 a3) S-1
        # see INDEXING.md for details

        if not all([x.is_integer for x in bravais_vectors]):
            raise TypeError("Bravais vectors must be integers")

        bravais_vectors = Matrix(bravais_vectors)

        if bravais_vectors.is_diagonal():
            D = bravais_vectors
            S = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            T = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            D, S, T = smith_normal_decomp(bravais_vectors, ZZ)
            signs = [ 1 if d>0 else -1 for d in D.diagonal()]
            signs = Matrix.diag(*signs)
            D *= signs
            T *= signs
            print(D,S,T)

            assert S * bravais_vectors * T == D

        self.periodicity = [int(x) for x in D.diagonal()]
        self.primitive = reshape_primitive_cell(primitive_suggestion, S.inv())

    def add_atom(self, a: Atom):
        J_raw = self.hash_tuple(*self.cell_index(a.xyz))
        self.atom_lookup[J_raw] = len(self.atoms)
        self.atoms.append(a)

    def _populate_atoms(self):
        for sl in self.primitive.sublattices:
            for ix, iy, iz in self.enumerate_primitives():
                delta = self.primitive.lattice_vectors @ Matrix(
                        [ix, iy, iz])
                self.add_atom(
                        Atom(sl.sl_name, sl.xyz + delta)
                        )


    def _populate_bonds(self):
        for a in self.primitive.sublattices:
            for ix, iy, iz in self.enumerate_primitives():
                X0 = self.primitive.lattice_vectors @ Matrix(
                        [ix, iy, iz])
                for b in a.bonds_from:
                    J_from = self.as_linear_idx(X0 + a.xyz)
                    J_to = self.as_linear_idx(X0 + a.xyz + b.bond_delta)

                    self.bonds.append(Bond(
                        from_idx=J_from,
                        to_idx=J_to,
                        bond_delta=b.bond_delta,
                        color=b.color
                        ))

    def cell_index(self, xyz: Matrix):
        '''
        Takes in the xyz poisition, returns a site index
        Note: after rolling, all xyz must have the form
        (A1 A2 A3)m + dr
        where m \\in Z^3, dr is strictly within the A1 A2 A2 unit cell
        Each SL has an index 
        Strategy: 
            0. roll xyz using b1 b2 b3
            1. decide which primitive unit cell we are in by solving (A1 A2 A3)m = xyz
            2. successively subtract off the primitive-cell positions, decide which is best
        '''
        xyz = self.wrap_coordinate(xyz)
        m = self.primitive.lattice_vectors.inv() @ xyz
        # intra-cell dr
        sl_idx = None
        cell_idx = Matrix([floor(x) for x in m])
        dr = self.primitive.lattice_vectors@(m - cell_idx)
        cell_idx = [int(j) for j in cell_idx]
        for j, sl in enumerate(self.primitive.sublattices):
            if (sl.xyz - dr).norm() == 0:
                sl_idx = j

        if sl_idx is None:
            print(self.lattice_vectors)
            print(f"xyz = {xyz}\nm={m}\ncell_idx={cell_idx}\ndr = {dr}")
            print([[x for x in s.xyz] for s in self.atoms])
            raise Exception(
                f"Position {xyz} does not appear to lie on the lattice")

        for j in range(3):
            assert (cell_idx[j] >= 0 and cell_idx[j] < self.periodicity[j])
        return (cell_idx, sl_idx)

    def delete_atom_at_idx(self, idx):
        del self.atoms[idx]
        to_remove = []
        for j, b in enumerate(self.bonds):
            if b.from_idx == idx or b.to_idx == idx:
                to_remove.append(j)

        for mu, j in enumerate(to_remove):
            del self.bonds[j-mu]

        # shuffle indices
        for b in self.bonds:
            if b.from_idx > idx:
                b.from_idx -= 1
            if b.to_idx > idx:
                b.to_idx -= 1

        for J_raw in self.atom_lookup:
            if self.atom_lookup[J_raw] is None:
                continue
            elif self.atom_lookup[J_raw] == idx:
                self.atom_lookup[J_raw] = None
            elif self.atom_lookup[J_raw] > idx:
                self.atom_lookup[J_raw] -= 1

        self.shape_hash_str += f'd{idx}'
        return self

    def hash_tuple(self, cell_idx, sl_idx):
        N = self.periodicity
        J = cell_idx[2] + N[2]*(cell_idx[1] +
                                   N[1]*(cell_idx[0] + N[0]*sl_idx))
        return J

    def as_linear_idx(self, xyz: Matrix):
        cell_idx, sl_idx = self.cell_index(xyz)
        return self.atom_lookup[
                self.hash_tuple(cell_idx, sl_idx)
                ]

    @property
    def num_atoms(self):
        return len(self.atoms)

    def wrap_coordinate(self, xyz):
        return _wrap_coordinate(self.lattice_vectors, xyz)


# INPUT OUTPUT

def listify(X):
    return [float(x) for x in X]


def to_dict(lat: Lattice):
    output = {}
    output["primitive"] = {}
    a = lat.primitive.lattice_vectors
    output["primitive"]["lattice_vectors"] = {
        'a0': listify(a[:, 0]),
        'a1': listify(a[:, 1]),
        'a2': listify(a[:, 2])
    }

    A = lat.lattice_vectors
    output["lattice_vectors"] = {
        'A0': listify(A[:, 0]),
        'A1': listify(A[:, 1]),
        'A2': listify(A[:, 2])
    }
    output["atoms"] = [
        {
            'xyz': listify(a.xyz),
            'sl': a.sl_name
        }
        for a in lat.atoms]
    output["bonds"] = [{
        'from_idx': b.from_idx,
        'to_idx': b.to_idx,
        'bond_delta': listify(b.bond_delta),
        'color': b.color
    }
        for b in lat.bonds]
    return output


def from_dict(data: dict, primitive_spec: PrimitiveCell):
    # Extract the primitive lattice vectors and Bravais vectors
    a = from_cols(*data["primitive"]["lattice_vectors"].values())
    primitive_cell = PrimitiveCell(a)
    for sl in primitive_spec.sublattices:
        primitive_cell.add_sublattice(sl.sl_name, sl.xyz)

    A = from_cols(*data["lattice_vectors"].values())

    # Create the Lattice object
    bv = primitive_cell.lattice_vectors.solve(A)

    lat = Lattice(primitive_suggestion=primitive_cell,
                  bravais_vectors=Matrix(3, 3, lambda i, j: int(bv[i, j])),
                  populate=False)

    assert len(lat.atoms) == 0
    for J, a in enumerate(data['atoms']):
        x = nsimplify(Matrix(a['xyz']), rational=True)
        lat.add_atom(Atom(sl_name=a['sl'], xyz=x))

    for b in data['bonds']:
        bd = Matrix(b['bond_delta'])
        bc = b['color'] if 'color' in b else 0
        lat.bonds.append(Bond(from_idx=b['from_idx'],
                              to_idx=b['to_idx'],
                              bond_delta=bd,
                              color=bc))

    # make sure we loaded things right
    for J, a in enumerate(lat.atoms):
        jj = lat.as_linear_idx(a.xyz)
        assert jj == J, f"Issue at idx {J} -> {jj}"

    return lat


## SYMMETRY OPERATIONS
def spg_op_as_perm(lat: Lattice, rotation, translation):
    '''
    Applies x -> Rx + t and returns the corresponding translation
    Assumes that both are in reduced units
    '''
    perm = []
    rotation = Matrix(rotation)
    translation = Matrix(translation)
    for (orig_idx, a) in enumerate(lat.atoms):
        transl_idx = lat.as_linear_idx(lat.lattice_vectors * (rotation * lat.lattice_vectors.inv() * a.xyz + translation))
        perm.append(transl_idx)
    return np.array(perm)

def get_symmetry(lat: Lattice):
    cell_dbl = np.array(lat.lattice_vectors, dtype=np.float64).T
    atom_positions = np.array([
        [float(xx) for xx in (lat.lattice_vectors.inv() * a.xyz)]
        for a in lat.atoms], dtype=np.float64)
    return spglib.get_symmetry((cell_dbl, atom_positions, np.ones(lat.num_atoms)))


def all_spg_perms(lat: Lattice):
    dataset = get_symmetry(lat)
    ops = []
    for r, t in zip(dataset['rotations'], dataset['translations']):
        ops.append(spg_op_as_perm(lat, r, t))

    return np.unique(ops, axis=0)


#def get_transl_generators(lat: Lattice):
#    """
#    Returns a list of gnerators for the three obvious translational
#    symmetries
#    """
#    d = lat.lattice_vectors.shape[0]  # ndim
#
#    retval = []
#
#    for i in range(d):
#        perm = []
#        for (orig_idx, a) in enumerate(lat.atoms):
#            transl_idx = lat.as_linear_idx(
#                a.xyz + lat.primitive.lattice_vectors[:, i])
#            perm.append(transl_idx)
#
#        retval.append(perm)
#
#    return retval
#
#def get_inversion_perm(lat: Lattice, inversion_centre_xyz):
#    """
#    Uses a hand-fed inversion centre to generate an inversion permutation
#    """
#    inv_xyz = Matrix(inversion_centre_xyz)
#    perm = []
#    for (orig_idx, a) in enumerate(lat.atoms):
#        dx = a.xyz-inv_xyz
#        transl_idx = lat.as_linear_idx(inv_xyz - dx)
#        perm.append(transl_idx)
#
#    return perm
#
#def get_refl_perm(lat:Lattice, origin: Matrix, direction: Matrix):
#    """
#    Returns a permutation corresponding to (possibly trivial) reflection
#    in the planes normal to 'direction' passing
#    through 'origin'
#    """
#
#    perm = []
#
#    unit_direction = direction / direction.norm()
#    for (orig_idx, a) in enumerate(lat.atoms):
#        relpos = a.xyz - origin
#        # project on to direction
#        delta = relpos.dot(unit_direction) * unit_direction
#        transl_idx = lat.as_linear_idx(
#            a.xyz - 2*delta)
#        perm.append(transl_idx)
#
#    return perm
#
#def get_rot_perm(lat:Lattice, origin: Matrix, rot_mat: Matrix):
#    """
#    Returns the permutation of the site indices associated with a global
#    rotation by 2π/3
#    """
#    perm = []
#
#    for (orig_idx, a) in enumerate(lat.atoms):
#        relpos = a.xyz - origin
#        # project on to direction
#        delta = rot_mat.dot(relpos)
#        transl_idx = lat.as_linear_idx(a.xyz + delta)
#        perm.append(transl_idx)
#
#    return perm
#
