from uint128 import UInt128

# import itertools
def as_mask(spin_indices: list) -> int:
    mask = UInt128(0)
    for j in spin_indices:
        mask |= UInt128(1) << j
    return mask


def make_state(spin_indices: list, state: int):
    # state = n-bit integer
    mask = UInt128(0)
    i = 0
    for spin_idx in spin_indices:
        mask |= ((state & (UInt128(1) << i)) >> i) << spin_idx
        i += 1
    return mask


# def cartesian_product(*arrays):
#     la = len(arrays)
#     dtype = np.result_type(*arrays)
#     arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
#     for i, a in enumerate(np.ix_(*arrays)):
#         arr[...,i] = a
#     return arr.reshape(-1, la)


def bitperm(perm: list,  x: UInt128):
    # does the specified permutation on the bits of x
    y = UInt128(0)
    for i in range(len(perm)):
        y |= ((x & (UInt128(1)<<i)) >> i) << perm[i]

    return y
