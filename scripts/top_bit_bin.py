import h5py
import numpy as np
import argparse


parser = argparse.ArgumentParser(
        description="A utility for ensuring that partitioning a certain UInt128 basis by the top N bits distributes the work evenly")

parser.add_argument("file", help="the HDF5 file")
parser.add_argument("--sector", help="the dataset name within the HDF file to read", default="basis")
parser.add_argument("--min_n", help="minimum value of n to start from (number of bits to mask)",
                    type=int,
                    default=1)
parser.add_argument("--max_n", help="maximum value of n to stop at (number of bits to mask)",
                    type=int,
                    default=128)

args = parser.parse_args()



file = h5py.File(args.file)
dset = file[args.sector]
total_elems = dset.shape[0]
local_size = 1
mask = np.array([2**64 - 1, 2**64 - 1], dtype=np.uint64)
n = 0

# Initialize mask to all 1s
mask = np.array([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)

# Skip to min_n by shifting the mask appropriately
while n < args.min_n:
    carry = (mask[0] & 0x8000000000000000) >> 63
    mask[1] = ((mask[1] << 1) | carry) & 0xFFFFFFFFFFFFFFFF
    mask[0] = (mask[0] << 1) & 0xFFFFFFFFFFFFFFFF
    n += 1


n_bins=100

while n <= args.max_n and n_bins > 1:

    print("=" * 100)
    print("n = %d" % n)
    print("mask: %016x %016x" % (mask[1], mask[0]))

    seen = {}
    
    for j, row in enumerate(dset):
        x = (row[1] & mask[1],row[0] & mask[0])
        if x in seen:
            seen[x] += 1
        else:
            seen[x] = 1

    n_bins = len(seen)
    mean_size = 1.0 * total_elems / n_bins
    mean2 = 0.0
    min_s = total_elems
    max_s = 0
    for v in seen.values():
        mean2 += v * v
        max_s = max(max_s, v)
        min_s = min(min_s, v)
    
    sigma_s = np.sqrt(mean2 / n_bins - mean_size ** 2)
    print("%d bins" % n_bins)
    print("bin size: ", mean_size, "±", sigma_s, " (%.2f %%)" % (100 * sigma_s/mean_size) )
    print("min bin: ", min_s, "; max bin: ", max_s)
    
    local_size = mean_size
    
    # Shift mask left by 1 for next iteration
    carry = (mask[0] & 0x8000000000000000) >> 63
    mask[1] = (mask[1] << 1) | carry
    mask[0] = (mask[0] << 1)
    n += 1
