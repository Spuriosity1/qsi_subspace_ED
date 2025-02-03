# import itertools
def as_mask(spin_indices) -> int:
    mask = 0
    for j in spin_indices:
        mask |= 1 << j
    return mask

def make_state(spin_indices: list, state:int):
    # state = n-bit integer
    mask = 0
    for i, spin_idx in enumerate(spin_indices):
        mask |= ((state & (1 << i)) >> i) << spin_idx 
    return mask


# def cartesian_product(*arrays):
#     la = len(arrays)
#     dtype = np.result_type(*arrays)
#     arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
#     for i, a in enumerate(np.ix_(*arrays)):
#         arr[...,i] = a
#     return arr.reshape(-1, la)


def bitperm(perm: list,  x: int):
    # does the specified permutation on the bits of x
    y = 0
    for i in range(len(perm)):
        y |= ((x & (1<<i)) >> i) << perm[i]

    return y

