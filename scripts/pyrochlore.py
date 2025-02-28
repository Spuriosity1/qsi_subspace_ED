import lattice
# import visual
# import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix
import json

disp = [
    Matrix(v) for v in [
        [0, 0, 0],
        [0, 2, 2],
        [2, 0, 2],
        [2, 2, 0]
    ]]

tetra_pos = Matrix([1, 1, 1])

plaqt = [
    [Matrix(x) for x in [
        [0, -2, 2],
        [2, -2, 0],
        [2, 0, -2],
        [0, 2, -2],
        [-2, 2, 0],
        [-2, 0, 2]]],
    [Matrix(x) for x in [
        [0, 2, -2],
        [2, 2, 0],
        [2, 0, 2],
        [0, -2, 2],
        [-2, -2, 0],
        [-2, 0, -2]]],
    [Matrix(x) for x in [
        [0, -2, -2],
        [-2, -2, 0],
        [-2, 0, 2],
        [0, 2, 2],
        [2, 2, 0],
        [2, 0, -2]]],
    [Matrix(x) for x in [
        [0, 2, 2],
        [-2, 2, 0],
        [-2, 0, -2],
        [0, -2, -2],
        [2, -2, 0],
        [2, 0, 2]]]
]

plaq_locs = [Matrix(x) for x in
             [[4, 4, 4], [4, 2, 2], [2, 4, 2], [2, 2, 4]]]


class Ring:
    def __init__(self, xyz, sl, members):
        self.xyz = xyz
        self.sl = sl
        self.members = members


class Tetra:
    def __init__(self, xyz, sl, members):
        self.xyz = xyz
        self.sl = sl
        self.members = members


def get_ringflips(lat: lattice.Lattice, sl=[0, 1, 2, 3], include_partial=False):
    if not hasattr(sl, "__iter__"):
        sl = [sl]

    if include_partial:
        retval = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    else:
        retval = []

    for ix, iy, iz in lat.enumerate_primitives():
        dx = lat.primitive.lattice_vectors @ Matrix([ix, iy, iz])
        for mu in sl:
            plaq_sl_pos = plaq_locs[mu]
            plaq_pos = lat.wrap_coordinate(plaq_sl_pos + dx)
            spin_members = [lat.as_linear_idx(plaq_pos + x)
                            for x in plaqt[mu]]
            n_missing = sum(j is None for j in spin_members)

            rr = Ring(plaq_pos, mu, spin_members)
            if include_partial:
                retval[n_missing].append(rr)
            elif n_missing == 0:
                retval.append(rr)
            

    return retval



def get_tetras(lat: lattice.Lattice):
    up = []  # indices of up tetras
    dn = []  # indices of down tetras
    for ix in range(lat.periodicity[0]):
        for iy in range(lat.periodicity[1]):
            for iz in range(lat.periodicity[2]):
                dx = lat.primitive.lattice_vectors @ Matrix([ix, iy, iz])
                up_idx = [lat.as_linear_idx(dx + delta) for delta in disp]
                dn_idx = [lat.as_linear_idx(dx - delta) for delta in disp]
                up.append(Tetra(dx + tetra_pos, 0, up_idx))
                dn.append(Tetra(dx - tetra_pos, 1, dn_idx))

    return up, dn

primitive = lattice.PrimitiveCell([[0, 4, 4],
                                   [4, 0, 4],
                                   [4, 4, 0]])
sublat = [primitive.add_sublattice(str(j), disp[j]) for j in range(4)]

sublat_pairs = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 3),
    (3, 1)
]

for c, (i,j) in enumerate(sublat_pairs):
    delta = disp[j] - disp[i]
    primitive.add_bond(sublat[i],  delta, c)
    primitive.add_bond(sublat[i], -delta, c)

# TODO change API to support making these difernet bond colours


def export_json(lat: lattice.Lattice, filename: str, include_partial=False):
    output = lattice.to_dict(lat)
    t_up, t_dn = get_tetras(lat)
    output["tetrahedra"] = []
    output["tetrahedra"] += [{
        'xyz': lattice.listify(t.xyz),
        'sl': 0,
        'member_spin_idx': t.members
        } for t in t_up]
    output["tetrahedra"] += [{
        'xyz': lattice.listify(t.xyz),
        'sl': 1,
        'member_spin_idx': t.members
        } for t in t_dn]
    if include_partial:
        output["rings"] = [
            [{
                'xyz': lattice.listify(r.xyz),
                'sl': r.sl,
                'member_spin_idx': r.members
            } for r in rr]
            for rr in get_ringflips(lat).values()
        ]
    else:
        output["rings"] = [{
            'xyz': lattice.listify(r.xyz),
            'sl': r.sl,
            'member_spin_idx': r.members
        } for r in get_ringflips(lat)]

    with open(filename, 'w') as f:
        json.dump(output, f)


def import_json(name:str):
    with open(name, 'r') as f:
        return lattice.from_dict(json.load(f), primitive)


