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

vol_delta = [
    Matrix(v) for v in [
        [-1, -1, -1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1]
    ]]


plaq_locs = [Matrix(x) for x in
             [[4, 4, 4], [4, 2, 2], [2, 4, 2], [2, 2, 4]]]

vol_locs = [Matrix(x) for x in ([3,3,3], [5,5,5])]

class Tetra:
    def __init__(self, xyz, sl, members):
        self.xyz = xyz
        self.sl = sl
        self.members = members

    def __repr__(self):
        return f"Tetra at {self.xyz} {self.sl}, members {self.members}"


class Ring:
    def __init__(self, xyz, sl, members, signs):
        self.xyz = xyz
        self.sl = sl
        self.members = members
        self.signs = signs

    def __repr__(self):
        return f"Ring at {self.xyz} {self.sl}, members {self.members}"


class Vol:
    def __init__(self, xyz, sl, member_ringid):
        self.xyz = xyz
        self.sl = sl
        self.member_plaq_ids = member_ringid

    def __repr__(self):
        return f"Vol at {self.xyz} {self.sl}, member ids {self.member_plaq_ids}"



def get_tetras(lat: lattice.Lattice):
    def _remove_None(it):
        return list(filter(lambda x: x is not None, it))

    up = []  # indices of up tetras
    dn = []  # indices of down tetras
    for ix in range(lat.periodicity[0]):
        for iy in range(lat.periodicity[1]):
            for iz in range(lat.periodicity[2]):
                dx = lat.primitive.lattice_vectors @ Matrix([ix, iy, iz])
                up_idx = [lat.as_linear_idx(dx + delta) for delta in disp]
                dn_idx = [lat.as_linear_idx(dx - delta) for delta in disp]
                up.append(Tetra(dx + tetra_pos, 0, _remove_None(up_idx)))
                dn.append(Tetra(dx - tetra_pos, 1, _remove_None(dn_idx)))

    return up, dn


def get_ringflips(lat: lattice.Lattice, sl=None, include_partial=False):
    if sl is None:
        sl = [0, 1, 2, 3]
    elif not hasattr(sl, "__iter__"):
        sl = [sl]

    retval = []

    for ix, iy, iz in lat.enumerate_primitives():
        dx = lat.primitive.lattice_vectors @ Matrix([ix, iy, iz])
        for mu in sl:
            plaq_sl_pos = plaq_locs[mu]
            plaq_pos = lat.wrap_coordinate(plaq_sl_pos + dx)
            spin_members = []
            signs = []
            sign = 1
            for x in plaqt[mu]:
                sign *= -1
                J = lat.as_linear_idx(plaq_pos + x)
                if J is not None:
                    spin_members.append(J)
                    signs.append(sign)

            rr = Ring(plaq_pos, mu, spin_members, signs)
            if include_partial:
                retval.append(rr)
            elif len(rr.members) == 6:
                retval.append(rr)

    return retval


def get_vols(lat: lattice.Lattice, sl=None, include_partial=False, rings=None):
    if sl is None:
        sl = [1,-1]
    elif not hasattr(sl, "__iter__"):
        sl = [sl]

    if rings is None:
        rings = get_ringflips(lat, None, include_partial)

    retval = []

    for ix, iy, iz in lat.enumerate_primitives():
        X0 = lat.primitive.lattice_vectors @ Matrix([ix, iy, iz])
        for i_eta, eta in enumerate(sl):
            vol_sl_pos = vol_locs[i_eta]
            vol_pos = lat.wrap_coordinate(X0 + vol_sl_pos)
            
            ring_ids = []
            for d in vol_delta:
                x = lat.wrap_coordinate(vol_pos - eta * d)
                ring_ids.append(
                        next(i for i,r in enumerate(rings) if r.xyz == x)
                        )

            retval.append( Vol(vol_pos, i_eta, ring_ids) )

    return retval


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


def export_json(lat: lattice.Lattice, filename: str):
    output = lattice.to_dict(lat)
    t_up, t_dn = get_tetras(lat)
    output["__version__"] = "1.1"
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
    output["rings"] = [{
        'xyz': lattice.listify(r.xyz),
        'sl': r.sl,
        'member_spin_idx': r.members,
        'signs': r.signs
    } for r in get_ringflips(lat, include_partial=True)]
    output["vols"] = [{
        'xyz': lattice.listify(v.xyz),
        'sl': v.sl,
        'member_plaq_idx': v.member_plaq_ids
    } for v in get_vols(lat, include_partial=True)]

    with open(filename, 'w') as f:
        json.dump(output, f)


def import_json(name:str):
    with open(name, 'r') as f:
        return lattice.from_dict(json.load(f), primitive)


